# RF de-embedding demo — round-trip verification

Pipeline: parse `raw/deemb_preext_cold.s2p`, `open.s2p`, `short.s2p` exactly as the app does (`parse_s2p`), extract pad capacitances with `step_open` and lead inductances/resistances with `step_short` (repo's own `tools/rf/ssm/helpers/deembed_math.py`), peel the extracted parasitics off the raw file with `peel_parasitics`, and compare the result to the original (pre-re-embedding) de-embedded file, `deembed_these/deembedded/4x10_left/deemb_preext_cold.s2p`.

Fixed parasitics used to build the demo set:

```
Cpbe = 22 fF, Cpbc = 5 fF, Cpce = 18 fF, Lb = 45 pH, Lc = 42 pH, Le = 6 pH, Rpb = Rpc = Rpe = 0 Ω
```

Extracted back from open.s2p / short.s2p:

```
Cpbe = 22.000000 fF
Cpce = 18.000000 fF
Cpbc = 5.000000 fF
Le = 6.000000 pH
Lb = 45.000000 pH
Lc = 42.000000 pH
Rpe = 1.178e-14 Ohm
Rpb = 5.027e-10 Ohm
Rpc = 5.023e-10 Ohm
```

## Result

```
Extracted from open.s2p:
  Cpbe = 22.000000 fF
  Cpce = 18.000000 fF
  Cpbc = 5.000000 fF
Extracted from short.s2p:
  Le = 6.000000 pH
  Lb = 45.000000 pH
  Lc = 42.000000 pH
  Rpe = 0.000000 Ohm
  Rpb = 0.000000 Ohm
  Rpc = 0.000000 Ohm

max |S_check - S_original| = 5.734e-10  (at index (np.int64(15), np.int64(1), np.int64(1)), f = 0.0848 GHz)
rms |S_check - S_original| = 2.139e-10
PASS (< 1e-6): True
```
