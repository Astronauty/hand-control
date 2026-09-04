"""Backend selector: return the DexPilot or AnyTeleop finger retargeter.

The AnyTeleop import is LAZY (inside the branch) so that the default DexPilot path — and
any environment without the `dex-retargeting` optional dependency installed — never
imports the AnyTeleop stack. This is what keeps the add-on fully separable: if
anyteleop/ is absent or its deps are not installed, `make_retargeter("dexpilot", ...)`
still works.
"""
from __future__ import annotations


def make_retargeter(name, model, **kwargs):
    """name: 'dexpilot' (default/hand-rolled) or 'anyteleop' (dex-retargeting)."""
    if str(name).lower() == "anyteleop":
        from anyteleop.retargeter import AnyTeleopRetargeter
        return AnyTeleopRetargeter(model, **kwargs)
    from teleop.dexpilot_retargeter import DexPilotRetargeter
    return DexPilotRetargeter(model, **kwargs)
