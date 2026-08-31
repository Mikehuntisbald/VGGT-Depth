# Environment dependency policy

Install the CUDA-enabled PyTorch build before any file in this directory. These
files intentionally do not declare `torch` or `torchvision`; upstream packages
must not downgrade the Blackwell-compatible build.

The recorded M0 baseline is PyTorch 2.10.0 with CUDA 12.8 wheels. Re-run
`tools/check_env.py` after every environment change.

