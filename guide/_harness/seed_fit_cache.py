"""Seed the fit cache with the extracted Cheng T model for the demo device.

The tuning walkthrough has to start from the model the extraction pages
actually produce, not from the simulator's defaults (which sit at a 374%
residual and make every screenshot look broken). Typing nineteen parameters
into the fine-tune panel costs a Streamlit rerun each and does not fit the
40 s call budget, so write them straight into the per-(DUT, model) cache
instead. `SSMModelTemplate` restores it on first render.

Values are the corrected set from the extraction walkthrough:
Re from the Z-parameter fit, Rb/Rc from Cold-HBT with Cex on its median, and
tauB/tauC from the transit-time fit rather than Cheng's analytic split.

    python3 guide/_harness/seed_fit_cache.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/tmp/app")

# Two keys, because the two pages cache under different names: Extraction uses
# the bare s2p basename, Simulation & Fitting prefixes "simfit_<model>_" onto
# the measured device's label (simulator.py: `_fit_fname`).
MODEL = "T"
FNAMES = [
    "deemb_preext_vce3.5_ib200u",
    f"simfit_{MODEL}_deemb_preext_vce3.5_ib200u",
]

# SI units, as fit_cache stores them.
PARAMS = {
    # pads and leads, from the open/short standards
    "Cpbe": 22e-15, "Cpce": 18e-15, "Cpbc": 5e-15,
    "Lb": 45e-12, "Lc": 42e-12, "Le": 6e-12,
    # access resistance: Rpe from the Z-parameter fit, Rpb/Rpc from Cold-HBT
    "Rpb": 46.7508, "Rpc": 19.0178, "Rpe": 64.3966,
    # extrinsic
    "Cbex": 51.2774e-15, "Cbcx": 146.7393e-15,
    # intrinsic
    "Rbi": 1103.1865, "Rbe": 138.8055, "Cbe": 121.1760e-15,
    "Rbc": 58781.6491, "Cbc": 484.2610e-15,
    "alpha0": 0.9901,
    # transit times from the tau_total fit, not Cheng's analytic split
    # (which returns a negative tauC for this device)
    "tauB": 19.0480e-12, "tauC": 0.1500e-12,
}


def main() -> int:
    from tools.rf.ssm.helpers.fit_cache import cache_path_str, get_fit, save_fit

    print("cache root:", cache_path_str())
    rc = 0
    for fname in FNAMES:
        ok = save_fit(fname, MODEL, PARAMS)
        back = get_fit(fname, MODEL) or {}
        bad = {k: (v, back.get(k)) for k, v in PARAMS.items()
               if abs(back.get(k, 0) - v) > abs(v) * 1e-9}
        print(f"{fname}: saved={ok} keys={len(back)} mismatches={bad or 'none'}")
        rc |= 0 if ok and not bad else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
