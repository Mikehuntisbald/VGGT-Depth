from tools.check_env import PROFILE_MODULES, parse_version


def test_cuda_version_ordering() -> None:
    assert parse_version("12.8") >= (12, 8)
    assert parse_version("13.0") >= (12, 8)
    assert parse_version("12.4") < (12, 8)


def test_profiles_are_separate() -> None:
    assert set(PROFILE_MODULES) == {"ffs", "vggt", "tsr"}
    assert "vggt_omega" in PROFILE_MODULES["vggt"]
    assert "open3d" in PROFILE_MODULES["ffs"]

