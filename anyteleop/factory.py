"""Backend selector: return the DexPilot or AnyTeleop finger retargeter.

The AnyTeleop import is LAZY (inside the branch) so that the default DexPilot path — and
any environment without the `dex-retargeting` optional dependency installed — never
imports the AnyTeleop stack. This is what keeps the add-on fully separable: if
anyteleop/ is absent or its deps are not installed, `make_retargeter("dexpilot", ...)`
still works.
"""
from __future__ import annotations


def make_retargeter(name, model, **kwargs):
    """name: 'dexpilot' (default/hand-rolled), 'anyteleop' (dex-retargeting), or
    'vwj' (the whole-arm-hand Vector-Wrist-Joint optimizer, arXiv:2506.09384).

    'vwj' returns a WHOLE-ROBOT retargeter (23-DOF arm+hand) with a different
    retarget() signature (it also needs the wrist target pose) — the controller
    detects it via the .whole_robot attribute and drives it on a separate path.
    """
    n = str(name).lower()
    if n == "anyteleop":
        from anyteleop.retargeter import AnyTeleopRetargeter
        return AnyTeleopRetargeter(model, **kwargs)
    if n == "vwj":
        from vwj.retargeter import VWJRetargeter
        # VWJRetargeter ignores dexpilot/anyteleop-only kwargs (eps, pinch_debounce,
        # type_override, …) via its **_ignored catch-all; pass through the ones it uses.
        return VWJRetargeter(model, **kwargs)
    if n == "vwj_upstream":
        # The UPSTREAM optimizer verbatim (local clone under third_party/), same whole-
        # robot interface — the A/B partner to the clean-room 'vwj'.
        from vwj.upstream_retargeter import VWJUpstreamRetargeter
        return VWJUpstreamRetargeter(model, **kwargs)
    from teleop.dexpilot_retargeter import DexPilotRetargeter
    return DexPilotRetargeter(model, **kwargs)
