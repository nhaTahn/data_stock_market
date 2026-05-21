# Best Committee F&B Bundle

Nguon chinh:

- run goc tot nhat: `confirm_vn100_fnb_committee_20260408_235445_r01`
- package advisor: `reports/advisor_shortlist/fnb_committee_best_20260409`

Noi dung bundle:

- `core/`
- config, metrics, predictions cua ban committee packaged de review nhanh
- `source_run_*` la file goc tu run committee trong training history
- `backtests/`
- tong hop committee, stability, threshold backtest, va doi chieu standalone
- `plots/`
- cac plot tong hop test/backtest tu package advisor
- `model/`
- artifact model hien co trong source run: `feature_scaler.npz`, `target_scaler.npz`, `linear_regression.joblib`
- `notes/`
- `advisor_summary.md` va `manifest.json`

Ghi chu:

- Source run hien khong co checkpoint `.keras` cho committee model trong thu muc run.
- Vi vay, phan `model/` chi gom artifact serialize dang ton tai thuc te.
- Metric committee tot nhat hien tai nam trong `backtests/committee_best_summary.json` va `notes/manifest.json`.
