# VN30 Point-in-Time Constituents

Use `vn30_historical.csv` to define the VN30 universe by date.

Required columns:

| column | meaning |
|---|---|
| `start_date` | first date the symbol is treated as active in VN30, inclusive |
| `end_date` | last date the symbol is treated as active in VN30, inclusive |
| `symbol` | ticker symbol, uppercase |

Example:

```csv
start_date,end_date,symbol
2020-02-03,2020-08-02,ROS
2020-02-03,2021-02-01,VCB
2020-08-03,2021-02-01,KDH
```

## How It Is Used

- `experiments/training/evaluate_vn30_meta_ensemble_simulation.py` filters train/validation rows using the point-in-time VN30 mask.
- `experiments/training/run_vn30_panel_probe.py --dynamic-universe` builds sequences on the full VN market first, then filters `(Date, code)` sequence rows using this file.
- Filtering after sequence construction avoids windows jumping across constituent-entry/exit gaps while still evaluating only active VN30 constituents on each target date.

## Run Dynamic VN30 Panel Probe

```bash
PYTHONPATH=. venv/bin/python experiments/training/run_vn30_panel_probe.py \
  --dynamic-universe \
  --constituents-csv market_lists/vn30_historical.csv \
  --run-name vn30_dynamic_panel_probe
```

## Notes

- The generated file currently uses a wide placeholder range (`2010-01-01` to `2030-12-31`) for symbols from `vn30.txt`, so it behaves like the current static VN30 basket.
- Replace placeholder rows with actual VN30 review windows when historical constituents are collected.
- Consecutive windows can be represented as either one continuous row or multiple review-period rows.
- The loader treats both `start_date` and `end_date` as inclusive.
- If you keep the placeholder static basket, report it as a survivor/static-VN30 sample and disclose survivorship bias.
- Holdout/test should stay closed until the final locked evaluation.
