"""AnyTeleop retargeting backend (separable add-on).

Wraps the published `dex-retargeting` library (the retargeting core of AnyTeleop,
Qin et al.) as a drop-in finger-retargeting backend for the LEAP hand, so it can be
compared head-to-head against this repo's hand-rolled DexPilot retargeter.

Self-contained by design: nothing here is imported unless an AnyTeleop run mode is
selected (see anyteleop.factory.make_retargeter). The existing DexPilot / contact-aware
code paths are untouched when this package is absent or unused. See anyteleop/README.md
for install (`uv pip install -e ".[anyteleop]"`) and removal.
"""
