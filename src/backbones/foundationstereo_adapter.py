"""Local loader and adapter for the official FoundationStereo checkpoint.

The project already contains :class:`backbones.ffs_adapter.FFSAdapter`, which
owns the padding, confidence and left/right-consistency contract used by the
training/evaluation code.  The official ``byran-wang/FoundationStereo``
release is a little different from Fast-FoundationStereo: its
``model_best_bp2.pth`` file is a training checkpoint (the weights live under
the ``model`` key), its forward method does not accept
``optimize_build_volume``, and the source tree imports ``flash_attn``
unconditionally.  This module keeps those compatibility details local and
does not modify anything below ``third_party/``.

Two released checkpoints are supported:

* ``11-33-40`` -- the compact DINOv2/DepthAnything ``vits`` variant;
* ``23-51-11`` -- the ``vitl`` variant (the historical config omits
  ``vit_size``; it is inferred from the state-dict dimensions).

The loader constructs the upstream model with a local, architecture-aware
``Feature``/``ContextNetDino`` replacement.  ``timm`` is asked for an
*uninitialised* EdgeNeXt so loading never attempts an unrelated network
download; all weights come from the supplied checkpoint.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ffs_adapter import FFSAdapter, FFSOutput

LOGGER = logging.getLogger(__name__)


class FoundationStereoLoadError(RuntimeError):
    """Raised when a FoundationStereo checkpoint cannot be loaded safely."""


class FoundationStereoDependencyError(FoundationStereoLoadError):
    """Raised when the optional upstream runtime dependencies are unavailable."""


class _ConfigDict(dict[str, Any]):
    """Small OmegaConf-compatible mapping used by the upstream model.

    FoundationStereo accesses options both as ``args.foo`` and ``args["foo"]``
    (and calls ``args.get``).  Depending on OmegaConf here would make the
    frozen-backbone adapter unnecessarily coupled to the training environment.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors normal attr error
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@dataclass(frozen=True)
class FoundationStereoLoadInfo:
    """Provenance and compatibility information attached to an adapter."""

    checkpoint: str
    architecture: str
    checkpoint_format: str
    strict: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    flash_attention: str
    flash_attention_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "architecture": self.architecture,
            "checkpoint_format": self.checkpoint_format,
            "strict": self.strict,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "flash_attention": self.flash_attention,
            "flash_attention_error": self.flash_attention_error,
        }


_DEFAULT_CONFIG: dict[str, Any] = {
    "corr_implementation": "reg",
    "corr_levels": 2,
    "corr_radius": 4,
    "hidden_dims": [128, 128, 128],
    "low_memory": False,
    "max_disp": 416,
    "mixed_precision": True,
    "n_downsample": 2,
    "n_gru_layers": 3,
    "valid_iters": 32,
    "vit_size": None,
}

_VIT_SPEC: dict[str, dict[str, Any]] = {
    # DINOv2 embedding dimension, DPT head ``features`` and the resulting
    # feature-map channels concatenated into EdgeNeXt x4.
    "vits": {
        "embed_dim": 384,
        "features": 64,
        "vit_output_dim": 32,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "embed_dim": 768,
        "features": 128,
        "vit_output_dim": 64,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "embed_dim": 1024,
        "features": 256,
        "vit_output_dim": 128,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def _as_path(value: str | os.PathLike[str] | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - project requires PyYAML
        raise FoundationStereoDependencyError(
            "FoundationStereo config loading requires PyYAML; install pyyaml"
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FoundationStereoLoadError(f"cannot read FoundationStereo config: {path}") from exc
    except Exception as exc:
        raise FoundationStereoLoadError(f"invalid FoundationStereo YAML: {path}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise FoundationStereoLoadError(f"FoundationStereo config must be a mapping: {path}")
    return dict(payload)


def _normalise_config(
    checkpoint: Path,
    config: Mapping[str, Any] | str | os.PathLike[str] | Path | None,
    *,
    vit_size: str | None,
    max_disp: int | None,
    iterations: int | None,
) -> _ConfigDict:
    values = dict(_DEFAULT_CONFIG)
    values["hidden_dims"] = list(_DEFAULT_CONFIG["hidden_dims"])

    if config is None:
        sibling = checkpoint.with_name("cfg.yaml")
        if sibling.is_file():
            values.update(_load_yaml(sibling))
    elif isinstance(config, Mapping):
        values.update(dict(config))
    else:
        values.update(_load_yaml(_as_path(config)))

    if vit_size is not None:
        values["vit_size"] = vit_size
    if max_disp is not None:
        values["max_disp"] = int(max_disp)
    if iterations is not None:
        values["valid_iters"] = int(iterations)

    architecture = values.get("vit_size")
    if architecture is not None:
        architecture = str(architecture).lower()
        if architecture not in _VIT_SPEC:
            raise FoundationStereoLoadError(
                f"unsupported FoundationStereo vit_size={architecture!r}; "
                f"expected one of {sorted(_VIT_SPEC)}"
            )
        values["vit_size"] = architecture

    try:
        hidden_dims = [int(v) for v in values["hidden_dims"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise FoundationStereoLoadError("hidden_dims must be a sequence of integers") from exc
    if len(hidden_dims) != 3 or any(v <= 0 for v in hidden_dims):
        raise FoundationStereoLoadError("FoundationStereo hidden_dims must contain 3 positive values")
    values["hidden_dims"] = hidden_dims
    for key in ("n_gru_layers", "n_downsample", "corr_levels", "corr_radius", "max_disp"):
        try:
            values[key] = int(values[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise FoundationStereoLoadError(f"FoundationStereo config field {key!r} is invalid") from exc
    if values["n_gru_layers"] != 3:
        # The released checkpoint has three recurrent levels.  A different
        # value changes the module topology and cannot be loaded safely.
        raise FoundationStereoLoadError(
            "FoundationStereo checkpoints require n_gru_layers=3; "
            f"got {values['n_gru_layers']}"
        )
    if values["max_disp"] <= 0:
        raise FoundationStereoLoadError("max_disp must be positive")
    return _ConfigDict(values)


def _torch_load(path: Path) -> Any:
    """Load a local training checkpoint across torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no ``weights_only`` argument.
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        # A checkpoint saved with CUDA storages should still be readable on a
        # CPU-only host.  Keep the original error as context for diagnostics.
        raise FoundationStereoLoadError(f"cannot deserialize checkpoint {path}: {exc}") from exc


def _extract_state_dict(payload: Any) -> tuple[Mapping[str, Tensor], str]:
    """Extract the model state from dict, state-dict, or serialized-module forms."""

    if isinstance(payload, nn.Module):
        return payload.state_dict(), "serialized_module"
    if not isinstance(payload, Mapping):
        raise FoundationStereoLoadError(
            "FoundationStereo checkpoint must be a mapping containing 'model' "
            f"or a serialized nn.Module, got {type(payload)!r}"
        )

    state: Any = None
    fmt = "state_dict"
    for key in ("model", "state_dict", "model_state_dict", "weights"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            fmt = f"dict[{key}]"
            break
    if state is None and payload and all(isinstance(k, str) for k in payload):
        # A bare OrderedDict state-dict.
        if all(isinstance(v, Tensor) for v in payload.values()):
            state = payload
            fmt = "bare_state_dict"
    if state is None:
        raise FoundationStereoLoadError(
            "FoundationStereo checkpoint has no tensor state dict under 'model', "
            "'state_dict', 'model_state_dict' or 'weights'"
        )
    if not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
        raise FoundationStereoLoadError("FoundationStereo model state must map string keys to tensors")

    # DDP and a few training wrappers add one common prefix.  Strip it only
    # when every key has the prefix, avoiding accidental partial rewrites.
    keys = list(state)
    for prefix in ("module.", "model."):
        if keys and all(k.startswith(prefix) for k in keys):
            state = {k[len(prefix) :]: v for k, v in state.items()}
            fmt += f"+strip:{prefix[:-1]}"
            break
    return state, fmt


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except (TypeError, ValueError):
        return None


def infer_foundation_stereo_vit_size(
    state_dict: Mapping[str, Tensor],
    configured: str | None = None,
) -> str:
    """Infer ``vits``/``vitb``/``vitl`` from checkpoint dimensions.

    ``11-33-40/cfg.yaml`` explicitly says ``vits``; older ``23-51-11`` configs
    do not contain that field, so the DINOv2 patch-embedding or transformer
    dimensions are used as a deterministic fallback.
    """

    if configured is not None:
        value = str(configured).lower()
        if value not in _VIT_SPEC:
            raise FoundationStereoLoadError(
                f"unsupported configured vit_size={configured!r}; expected {sorted(_VIT_SPEC)}"
            )
        return value

    candidates = (
        "feature.dino.depth_anything.pretrained.patch_embed.proj.weight",
        "feature.dino.depth_anything.pretrained.blocks.0.attn.qkv.weight",
    )
    embed_dim: int | None = None
    for key in candidates:
        shape = _shape(state_dict.get(key))
        if shape is None:
            continue
        if key.endswith("patch_embed.proj.weight") and shape:
            embed_dim = int(shape[0])
            break
        if len(shape) >= 1:
            embed_dim = int(shape[0] // 3) if shape[0] % 3 == 0 else int(shape[-1])
            break
    if embed_dim is None:
        # Last-resort block count distinguishes the released vits/vitl pair.
        block_ids: list[int] = []
        marker = "feature.dino.depth_anything.pretrained.blocks."
        for key in state_dict:
            if marker not in key:
                continue
            tail = key.split(marker, 1)[1]
            token = tail.split(".", 1)[0]
            if token.isdigit():
                block_ids.append(int(token))
        if block_ids:
            return "vitl" if max(block_ids) >= 23 else "vits"
        raise FoundationStereoLoadError(
            "cannot infer FoundationStereo vit_size: DINOv2 patch/transformer keys are missing"
        )

    for name, spec in _VIT_SPEC.items():
        if int(spec["embed_dim"]) == embed_dim:
            return name
    raise FoundationStereoLoadError(
        f"cannot infer FoundationStereo vit_size from DINO embedding dimension {embed_dim}"
    )


def _fallback_flash_attn_func(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    **_: Any,
) -> Tensor:
    """API-compatible FlashAttention fallback implemented with PyTorch SDPA.

    FlashAttention uses ``[B, L, heads, head_dim]`` while SDPA uses
    ``[B, heads, L, head_dim]``.  The released model only requests inference
    (dropout zero); a small explicit einsum fallback keeps compatibility with
    torch versions predating ``scaled_dot_product_attention``.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("flash attention fallback expects [B,L,heads,head_dim] tensors")
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(q.shape[-1]))

    # The FoundationStereo code passes an unlimited window.  Respect finite
    # windows too, which makes this shim useful for direct callers/tests.
    attn_mask: Tensor | None = None
    left, right = window_size if window_size is not None else (-1, -1)
    if left >= 0 or right >= 0:
        q_len, k_len = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(q_len, device=q.device)[:, None]
        k_pos = torch.arange(k_len, device=q.device)[None, :]
        lower = q_pos - (left if left >= 0 else k_len)
        upper = q_pos + (right if right >= 0 else k_len)
        attn_mask = (k_pos >= lower) & (k_pos <= upper)
        attn_mask = attn_mask.to(dtype=torch.bool)

    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if sdpa is not None:
        kwargs: dict[str, Any] = {
            "attn_mask": attn_mask,
            "dropout_p": float(dropout_p),
            "is_causal": bool(causal) and attn_mask is None,
        }
        # ``scale`` was added after SDPA itself on a few torch releases.
        try:
            out = sdpa(q, k, v, scale=scale, **kwargs)
        except TypeError:  # pragma: no cover - exercised on old torch only
            out = sdpa(q * scale * math.sqrt(q.shape[-1]), k, v, **kwargs)
    else:  # pragma: no cover - old torch compatibility
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, -torch.inf)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -torch.inf)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out.transpose(1, 2).contiguous()


def _fallback_flash_attn_qkvpacked_func(
    qkv: Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs: Any,
) -> Tensor:
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError("packed flash attention fallback expects [B,L,3,heads,head_dim]")
    return _fallback_flash_attn_func(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        **kwargs,
    )


def _install_flash_attention_fallback(
    *,
    allow_fallback: bool,
) -> tuple[str, str | None, Any]:
    """Ensure importing upstream ``core.submodule`` works without flash_attn.

    Returns ``(mode, error, previous_module)``; the previous module is kept so
    callers can restore ``sys.modules`` after importing the upstream model.
    """

    previous = sys.modules.get("flash_attn")
    try:
        module = importlib.import_module("flash_attn")
        # Import can succeed while symbols are absent in an old package.
        if not hasattr(module, "flash_attn_func") or not hasattr(module, "flash_attn_qkvpacked_func"):
            raise ImportError("flash_attn module lacks required functions")
        return "flash_attn", None, previous
    except Exception as exc:
        if not allow_fallback:
            raise FoundationStereoDependencyError(
                "FoundationStereo requires flash_attn; install flash-attn or "
                "set allow_flash_fallback=True"
            ) from exc
        fallback = types.ModuleType("flash_attn")
        fallback.flash_attn_func = _fallback_flash_attn_func
        fallback.flash_attn_qkvpacked_func = _fallback_flash_attn_qkvpacked_func
        fallback.__dict__["__foundationstereo_fallback__"] = True
        sys.modules["flash_attn"] = fallback
        return "torch_sdpa_fallback", f"{type(exc).__name__}: {exc}", previous


def _restore_module(name: str, previous: Any) -> None:
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _import_upstream_foundation_stereo(
    repo_root: Path,
    *,
    allow_flash_fallback: bool,
) -> tuple[Any, str, str | None, Callable[[], None]]:
    """Import official FoundationStereo modules without polluting ``core``.

    The upstream source uses top-level imports (``from core...``), which can
    collide with the Fast-FoundationStereo checkout.  We temporarily remove
    those aliases, import the module, and restore the caller's aliases.  Class
    methods retain their original module globals after the import completes.
    """

    repo = _as_path(repo_root)
    if not (repo / "core" / "foundation_stereo.py").is_file():
        raise FoundationStereoLoadError(f"FoundationStereo source is missing: {repo}")

    flash_mode, flash_error, previous_flash = _install_flash_attention_fallback(
        allow_fallback=allow_flash_fallback
    )

    prefixes = ("core", "depth_anything", "dinov2")
    aliases = {
        name
        for name in sys.modules
        if name == "Utils"
        or any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    # Include the canonical names even when they have not been imported yet so
    # restoration is deterministic.
    aliases.update({"core", "Utils", "depth_anything", "dinov2"})
    previous = {name: sys.modules.get(name) for name in aliases}
    old_path = list(sys.path)
    for name in aliases:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("core.foundation_stereo")
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        sys.path[:] = old_path
        for name, value in previous.items():
            _restore_module(name, value)
        _restore_module("flash_attn", previous_flash)
        raise FoundationStereoDependencyError(
            "cannot import official FoundationStereo runtime; missing "
            f"dependency {missing!r}. Install third_party/FoundationStereo's "
            "environment requirements (flash_attn is handled automatically)."
        ) from exc
    except Exception as exc:
        sys.path[:] = old_path
        for name, value in previous.items():
            _restore_module(name, value)
        _restore_module("flash_attn", previous_flash)
        raise FoundationStereoDependencyError(
            f"cannot import official FoundationStereo runtime: {type(exc).__name__}: {exc}"
        ) from exc

    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        sys.path[:] = old_path
        # Remove any nested modules imported by the official checkout that
        # were not present before this scoped activation.
        for name in list(sys.modules):
            if name not in previous and (
                name == "Utils"
                or name == "flash_attn"
                or any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
            ):
                sys.modules.pop(name, None)
        for name, value in previous.items():
            _restore_module(name, value)
        _restore_module("flash_attn", previous_flash)

    # Keep the temporary top-level aliases alive until the model has been
    # constructed.  ``DepthAnythingFeature.__init__`` performs a runtime
    # ``from depth_anything.dpt import DepthAnything`` import.
    return module, flash_mode, flash_error, cleanup


def _configured_feature_class(upstream: Any, vit_size: str) -> type[nn.Module]:
    """Create the upstream ``Feature`` variant matching ``vit_size``."""

    spec = _VIT_SPEC[vit_size]
    base = upstream.Feature
    DepthAnythingFeature = upstream.DepthAnythingFeature
    BasicConv = upstream.BasicConv
    Conv2x_IN = upstream.Conv2x_IN
    ResidualBlock = upstream.ResidualBlock
    freeze_model = upstream.freeze_model
    get_resize_keep_aspect_ratio = upstream.get_resize_keep_aspect_ratio

    class ConfiguredFeature(base):
        def __init__(self) -> None:
            # Deliberately bypass ``base.__init__``: it hard-codes vitl and
            # requests pretrained EdgeNeXt weights from the network.
            nn.Module.__init__(self)
            import timm

            model = timm.create_model("edgenext_small", pretrained=False, features_only=False)
            self.stem = model.stem
            self.stages = model.stages
            chans = [48, 96, 160, 304]
            self.chans = chans

            self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
            self.deconv16_8 = Conv2x_IN(chans[2] * 2, chans[1], deconv=True, concat=True)
            self.deconv8_4 = Conv2x_IN(chans[1] * 2, chans[0], deconv=True, concat=True)
            output_dim = chans[0] * 2 + int(spec["vit_output_dim"])
            self.conv4 = nn.Sequential(
                BasicConv(output_dim, output_dim, kernel_size=3, stride=1, padding=1, norm="instance"),
                ResidualBlock(output_dim, output_dim, norm_fn="instance"),
                ResidualBlock(output_dim, output_dim, norm_fn="instance"),
            )

            self.dino = DepthAnythingFeature(encoder=vit_size)
            self.dino = freeze_model(self.dino)
            self.patch_size = 14
            self.d_out = [output_dim, chans[1] * 2, chans[2] * 2, chans[3]]

        def forward(self, x: Tensor) -> tuple[list[Tensor], Tensor]:
            batch, _, height, width = x.shape
            divider = math.lcm(self.patch_size, 16)
            h_resize, w_resize = get_resize_keep_aspect_ratio(
                height, width, divider=divider, max_H=1344, max_W=1344
            )
            x_in = F.interpolate(x, size=(h_resize, w_resize), mode="bicubic", align_corners=False)
            self.dino = self.dino.eval()
            with torch.no_grad():
                output = self.dino(x_in)
            vit_feat = output["out"]
            vit_feat = F.interpolate(vit_feat, size=(height // 4, width // 4), mode="bilinear", align_corners=True)

            x = self.stem(x)
            x4 = self.stages[0](x)
            x8 = self.stages[1](x4)
            x16 = self.stages[2](x8)
            x32 = self.stages[3](x16)
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = torch.cat([x4, vit_feat], dim=1)
            x4 = self.conv4(x4)
            return [x4, x8, x16, x32], vit_feat

    ConfiguredFeature.__name__ = "Feature"
    ConfiguredFeature.__qualname__ = "Feature"
    return ConfiguredFeature


def _configured_context_class(upstream: Any, vit_size: str) -> type[nn.Module]:
    """Create a ContextNetDino whose x4 input matches the selected DINO head."""

    spec = _VIT_SPEC[vit_size]
    base = upstream.ContextNetDino
    BasicConv = upstream.BasicConv

    class ConfiguredContextNetDino(base):
        def __init__(self, output_dim: Any = (128, 128, 128), norm_fn: str = "batch", downsample: int = 3) -> None:
            super().__init__(output_dim=output_dim, norm_fn=norm_fn, downsample=downsample)
            self.vit_feat_dim = int(spec["vit_output_dim"])
            if self.vit_feat_dim != 128:
                # Keep the same parameter names (conv2.conv/bn) while fixing
                # the input channel count for vits/vitb checkpoints.
                self.conv2 = BasicConv(
                    128 + self.vit_feat_dim,
                    128,
                    kernel_size=3,
                    padding=1,
                )

    ConfiguredContextNetDino.__name__ = "ContextNetDino"
    ConfiguredContextNetDino.__qualname__ = "ContextNetDino"
    return ConfiguredContextNetDino


class _FoundationStereoCallShim(nn.Module):
    """Translate the official forward signature to the FFSAdapter contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = model

    @property
    def classifier(self) -> nn.Module:
        return self.inner.classifier  # type: ignore[attr-defined]

    @property
    def update_block(self) -> nn.Module:
        return self.inner.update_block  # type: ignore[attr-defined]

    @property
    def args(self) -> Any:
        return getattr(self.inner, "args", None)

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        iters: int,
        test_mode: bool,
        optimize_build_volume: str = "pytorch1",
        **kwargs: Any,
    ) -> Tensor:
        if optimize_build_volume not in {"pytorch1", "native", "default"}:
            raise ValueError(
                "official FoundationStereo only provides the PyTorch volume "
                f"backend; got optimize_build_volume={optimize_build_volume!r}"
            )
        result = self.inner.forward(
            left,
            right,
            iters=int(iters),
            test_mode=bool(test_mode),
            low_memory=bool(kwargs.get("low_memory", False)),
            init_disp=kwargs.get("init_disp"),
        )
        if not isinstance(result, Tensor):
            raise TypeError(
                "official FoundationStereo test_mode=True must return a disparity Tensor, "
                f"got {type(result)!r}"
            )
        return result


class FoundationStereoAdapter(FFSAdapter):
    """FFSAdapter-compatible wrapper around official FoundationStereo."""

    def __init__(
        self,
        model: nn.Module,
        *,
        load_info: FoundationStereoLoadInfo | None = None,
        spatial_scale: float = 1.0,
        iterations: int = 32,
        volume_backend: str = "pytorch1",
        update_tau_input_px: float = 1.0,
        lr_sigma_lr_px: float = 1.0,
    ) -> None:
        if not isinstance(model, _FoundationStereoCallShim):
            model = _FoundationStereoCallShim(model)
        super().__init__(
            model,
            spatial_scale=spatial_scale,
            iterations=iterations,
            volume_backend=volume_backend,
            update_tau_input_px=update_tau_input_px,
            lr_sigma_lr_px=lr_sigma_lr_px,
        )
        self.load_info = load_info

    @property
    def foundation_model(self) -> nn.Module:
        """Return the underlying official model (without the call shim)."""

        model = self.model
        return model.inner if isinstance(model, _FoundationStereoCallShim) else model

    @torch.inference_mode()
    def forward(
        self,
        left_rgb: Tensor,
        right_rgb: Tensor,
        *,
        right_left_check: bool = False,
    ) -> FFSOutput:
        output = super().forward(left_rgb, right_rgb, right_left_check=right_left_check)
        if self.load_info is None:
            return output
        metadata = dict(output.metadata)
        metadata["foundationstereo"] = self.load_info.as_dict()
        return replace(output, metadata=metadata)


def load_foundation_stereo(
    checkpoint: str | os.PathLike[str] | Path,
    *,
    config: Mapping[str, Any] | str | os.PathLike[str] | Path | None = None,
    repo_root: str | os.PathLike[str] | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    vit_size: str | None = None,
    max_disp: int | None = None,
    iterations: int | None = None,
    spatial_scale: float = 1.0,
    volume_backend: str = "pytorch1",
    update_tau_input_px: float = 1.0,
    lr_sigma_lr_px: float = 1.0,
    strict: bool = True,
    allow_flash_fallback: bool = True,
) -> FoundationStereoAdapter:
    """Load a byran-wang FoundationStereo training checkpoint.

    Args:
        checkpoint: Path to ``model_best_bp2.pth`` (dict with ``model`` key).
        config: Optional YAML path or mapping.  By default, a sibling
            ``cfg.yaml`` is read when present.
        repo_root: Official source checkout.  Defaults to the local
            ``third_party/FoundationStereo`` directory.
        device: Target device.  ``None`` selects CUDA when available and CPU
            otherwise.
        dtype: Optional floating-point model dtype.  Inputs remain in [0,255]
            and are converted by the caller as needed.
        strict: Pass through to ``load_state_dict``; strict loading is the
            default so architecture/checkpoint mistakes fail closed.
        allow_flash_fallback: Install the SDPA fallback when flash-attn is
            absent or ABI-incompatible.
    """

    checkpoint_path = _as_path(checkpoint)
    if not checkpoint_path.is_file():
        raise FoundationStereoLoadError(f"FoundationStereo checkpoint does not exist: {checkpoint_path}")
    if repo_root is None:
        repo = Path(__file__).resolve().parents[2] / "third_party" / "FoundationStereo"
    else:
        repo = _as_path(repo_root)

    payload = _torch_load(checkpoint_path)
    state_dict, checkpoint_format = _extract_state_dict(payload)
    # Release the optimizer/scaler/RNG payload before constructing the model;
    # 11-33-40 and 23-51-11 are multi-gigabyte files.
    del payload

    cfg = _normalise_config(
        checkpoint_path,
        config,
        vit_size=vit_size,
        max_disp=max_disp,
        iterations=iterations,
    )
    architecture = infer_foundation_stereo_vit_size(state_dict, cfg.get("vit_size"))
    cfg["vit_size"] = architecture

    upstream, flash_mode, flash_error, cleanup_upstream = _import_upstream_foundation_stereo(
        repo,
        allow_flash_fallback=allow_flash_fallback,
    )

    # Build the exact architecture selected by the checkpoint.  The global
    # names are patched only while the upstream constructor resolves them;
    # third_party files remain byte-for-byte untouched.
    old_feature = upstream.Feature
    old_context = upstream.ContextNetDino
    upstream.Feature = _configured_feature_class(upstream, architecture)
    upstream.ContextNetDino = _configured_context_class(upstream, architecture)
    try:
        try:
            model = upstream.FoundationStereo(cfg)
        except Exception as exc:
            raise FoundationStereoLoadError(
                f"failed to construct FoundationStereo ({architecture}): {type(exc).__name__}: {exc}"
            ) from exc
    finally:
        upstream.Feature = old_feature
        upstream.ContextNetDino = old_context
        cleanup_upstream()

    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise FoundationStereoLoadError(
            "FoundationStereo checkpoint is incompatible with the selected "
            f"{architecture} architecture: {exc}"
        ) from exc

    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise FoundationStereoLoadError("CUDA was requested for FoundationStereo but is unavailable")
    # CUDA autocast is harmlessly disabled on CPU in modern torch, but turning
    # it off avoids noisy warnings and keeps CPU smoke tests deterministic.
    if target_device.type != "cuda":
        cfg["mixed_precision"] = False
        model.args.mixed_precision = False
    model = model.to(device=target_device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    info = FoundationStereoLoadInfo(
        checkpoint=str(checkpoint_path),
        architecture=architecture,
        checkpoint_format=checkpoint_format,
        strict=bool(strict),
        missing_keys=missing,
        unexpected_keys=unexpected,
        flash_attention=flash_mode,
        flash_attention_error=flash_error,
    )
    adapter = FoundationStereoAdapter(
        _FoundationStereoCallShim(model),
        load_info=info,
        spatial_scale=spatial_scale,
        iterations=int(iterations if iterations is not None else cfg.get("valid_iters", 32)),
        volume_backend=volume_backend,
        update_tau_input_px=update_tau_input_px,
        lr_sigma_lr_px=lr_sigma_lr_px,
    )
    return adapter


def infer_foundation_stereo(
    left_rgb: Tensor,
    right_rgb: Tensor,
    *,
    adapter: FoundationStereoAdapter | FFSAdapter | None = None,
    checkpoint: str | os.PathLike[str] | Path | None = None,
    right_left_check: bool = False,
    **load_kwargs: Any,
) -> FFSOutput:
    """Run one FoundationStereo inference using an adapter or checkpoint.

    Supplying an already-loaded ``adapter`` is recommended for multiple Spring
    frames.  If omitted, ``checkpoint`` is loaded once for this call.
    """

    if adapter is None:
        if checkpoint is None:
            raise ValueError("infer_foundation_stereo requires adapter or checkpoint")
        adapter = load_foundation_stereo(checkpoint, **load_kwargs)
    if not isinstance(adapter, FFSAdapter):
        raise TypeError(f"adapter must be an FFSAdapter, got {type(adapter)!r}")
    return adapter(left_rgb, right_rgb, right_left_check=right_left_check)


# A descriptive alias for callers that prefer to make the adapter nature
# explicit.  Keep the short function name above as the stable public API.
load_foundation_stereo_adapter = load_foundation_stereo


__all__ = [
    "FoundationStereoAdapter",
    "FoundationStereoDependencyError",
    "FoundationStereoLoadError",
    "FoundationStereoLoadInfo",
    "infer_foundation_stereo",
    "infer_foundation_stereo_vit_size",
    "load_foundation_stereo",
    "load_foundation_stereo_adapter",
]
