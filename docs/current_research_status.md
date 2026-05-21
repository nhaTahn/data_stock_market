# Trang Thai Nghien Cuu Hien Tai

Updated: 2026-04-27

## Pham Vi

Tat ca ket qua hien tai chi dung train va in-sample validation:

- Train: den `2020-03-31`
- In-sample validation: `2020-04-01` den `2022-11-15`
- Khong dung out-sample/test de chon model, tune tham so, chon feature, hay chon router

Out-sample chi duoc mo khi da chot model/router cuoi cung.

## Ket Luan Ngan

Huong tot nhat hien tai khong phai la them nhieu feature hay doi window/loss lung tung. Ket qua cho thay edge chinh dang nam o cross-sectional ranking theo ngay va theo regime, dac biet la downtrend.

Model nen giu:

- Prediction anchor: `broad_signmag_prune_general_sector_full_20260424_r04`
- Trade challenger: `broad_signmag_prune_phase_ic_sector19_20260425_r09`
- Router dang dang tin nhat ve ranking: `sector19_down_up_anchor_else`

Khong nen tiep tuc:

- Giam window xuong 10
- Tang window len 20
- Doi objective sang `rel_score_sharp` lam objective chinh
- Doi objective sang `rel_score_weighted` lam objective chinh
- Them market/regime feature tong quat khi chua co chan doan IC ro hon
- Train rieng downtrend expert hoac soft-weight downtrend neu khong co A/B moi ro rang hon

## Active Candidates

| Vai tro | Run / Report | Ket qua chinh | Cach dung |
| --- | --- | --- | --- |
| Prediction anchor | `broad_signmag_prune_general_sector_full_20260424_r04` | Signmag validation `rel_score=+0.00534`, quartile equity `2.477` | Model standalone dang tin nhat cho prediction |
| Trade challenger | `broad_signmag_prune_phase_ic_sector19_20260425_r09` | Validation `rel_score` thap hon anchor, nhung quartile equity `2.674` | Khong thay anchor, chi dung lam challenger/router |
| Conservative ensemble | `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01` | `w_challenger=0.10`: validation `rel_score=+0.0058`, equity `3.260` | Neu can prediction-safe ensemble |
| Trade ensemble | `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01` | `w_challenger=0.75`: validation equity `4.546`, hit rate `59.1%` | Chi la validation-optimized trade clue, can can trong overfit |
| Train-selected router | `reports/router_train_selected/anchor_sector19_train_selected_20260426_r01` | Train-selected trade rules chi dat validation equity `3.44` | Dung de kiem tra overfit cua weight-grid |
| Cross-sectional IC router | `reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01` | `sector19_down_up_anchor_else`: mean IC `+0.0538`, t-stat `+5.51` | Huong nghien cuu chinh tiep theo |
| Rank objective offline | `reports/rank_objective_offline/anchor_sector19_rank_objective_20260427_r01` | `sector19_down_up_anchor_else`: val Spearman IC `+0.0539`, top-bottom equity `4.026` | Bang chung nen lam rank/portfolio sidecar |
| Train-selected rank router | `reports/rank_router_train_selected/anchor_sector19_rank_router_20260427_r01` | `train_rank_regime_ic_weight`: val quartile equity `4.250`, worst-year equity `1.472` | Trade-side improvement, khong thay prediction anchor |
| Stock reliability filter | `reports/stock_reliability_filter/anchor_sector19_stock_reliability_20260427_r01` | Best van la all-stocks; filter stock lam giam IC/equity | Negative result |
| Stock bias calibration | `reports/stock_bias_calibration/anchor_sector19_stock_bias_calibration_20260427_r01` | Bias calibration cai thien rel_score nhe cho router nhung giam IC/equity | Limited/khong active |

## Nhung Gi Da Lam

### 1. Don report va training run

Da chuyen huong report ve cac folder tong hop de doc nhanh hon thay vi mo qua nhieu file trong tung run.

Report nen doc truoc:

- `reports/feature_pruning/.../summary.md`
- `reports/router_analysis/.../summary.md`
- `reports/router_weight_grid/.../summary.md`
- `reports/router_train_selected/.../summary.md`
- `reports/cross_sectional_ic/.../summary.md`

Da archive cac run khong hieu qua vao:

- `data/processed/assets/data_info_vn/history/training_runs/_archive/ineffective_20260425_phase_research/`
- `data/processed/assets/data_info_vn/history/training_runs/_archive/ineffective_20260426_objective_window_smoke/`

### 2. Feature pruning tu broad signmag

Muc tieu ban dau la giam bot feature va sua `vingroup_momentum`: ten cu lam hieu sai thanh rieng Vingroup, trong khi y tuong dung la basket cac ma anh huong thi truong. Feature moi nen dung ten `market_leader_return` va chon leader theo thanh khoan/gia tri giao dich lich su.

Ket qua quan trong:

- `general_sector_full` la portable anchor tot nhat hien tai.
- Sector features tong quat tot hon viec hard-code VIN group.
- `phase_ic_sector19` co trade behavior tot nhung prediction metric khong du tot de thay anchor.
- Nhieu bien the compact hon co luc trade tot nhung `rel_score` khong on dinh.

Ket luan:

- Giu `general_sector_full` lam anchor.
- Giu `phase_ic_sector19` lam challenger, khong dung standalone lam prediction anchor.

### 3. Phase / cycle / regime analysis

Da tach hieu nang theo market regime de kiem tra y tuong "uptrend, downtrend, distribution thi feature nen khac nhau".

Ket qua quan trong:

- Anchor hoat dong tot hon trong `distribution`, `sideways`, va `recovery`.
- Anchor yeu trong `downtrend`.
- Sector19/challenger giup sua downtrend tot hon anchor.

Ket luan:

- Nen phat trien theo huong regime-specific expert/router.
- Khong nen chon chu ky bang mat.
- Nen de regime rule/IC chi ra pha nao model yeu.

Cap nhat sau follow-up 2026-04-27:

- Hard-filter LSTM downtrend expert chua dat; LSTM overfit va baseline/seed khong on dinh.
- Downtrend sidecar khong thang `sector19_down_anchor_else` du ro ve downtrend IC/equity.
- Soft downtrend weighting `1.5/2.0/3.0` lam giam `rel_score`/equity tong the va khong sua duoc downtrend IC.
- Ket luan moi: uu tien router/rank objective hon la train them downtrend expert.

### 4. Router va ensemble

Da thu hai nhom router:

- Validation weight-grid
- Train-selected router

Ket qua validation weight-grid:

| Candidate | Validation `rel_score` | Equity | Ghi chu |
| --- | ---: | ---: | --- |
| `w_challenger=0.10` | `+0.0058` | `3.260` | Tot nhat cho prediction trong weight-grid |
| `w_challenger=0.75` | `+0.0026` | `4.546` | Tot nhat cho trade equity nhung co rui ro overfit validation |
| Anchor | `+0.0053` | `3.241` | Benchmark |

Ket qua train-selected router:

| Candidate | Validation `rel_score` | Equity | Ghi chu |
| --- | ---: | ---: | --- |
| `train_regime_trade_weight` | `+0.0032` | `3.449` | Tot nhat khi rule duoc chon tren train |
| `train_global_trade_weight` | `+0.0026` | `3.438` | Cai thien trade nhe |
| Anchor | `+0.0053` | `3.241` | Prediction van tot nhat |

Ket luan:

- Weight-grid `w=0.75` nhin dep nhung co kha nang overfit validation.
- Train-selected router cai thien trade it hon nhieu, nen khong du de chot.
- Router co ich, nhung nen dua tren regime/IC hon la grid weight don gian.

### 5. Objective/window smoke

Da chay smoke batch:

`reports/feature_pruning/broad_signmag_prune_20260426_r02`

Ket qua:

| Case | Signmag val `rel_score` | Signmag equity | Ket luan |
| --- | ---: | ---: | --- |
| Anchor smoke, window 15, `rel_score` | `+0.0053` | `2.477` | Benchmark dung lai duoc |
| Window 20 | `+0.0032` | `1.356` | Kem hon anchor |
| `rel_score_weighted` | `+0.0028` | `1.708` | Kem hon anchor |
| Window 10 | `+0.0015` | `1.283` | Kem hon anchor |
| `rel_score_sharp` | `+0.0004` | `2.637` | Trade co tin hieu nhung prediction qua kem |

Ket luan:

- Window 15 hien la lua chon nen giu.
- Khong mo rong window 10/20.
- Khong doi objective chinh sang `rel_score_sharp` hoac `rel_score_weighted`.
- `rel_score_sharp` chi la clue cho trade-side/large-move, khong phai anchor objective.

### 6. Cross-sectional IC

Report:

`reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01`

Validation ranking:

| Candidate | Mean daily IC | t-stat | Positive days |
| --- | ---: | ---: | ---: |
| `sector19_down_up_anchor_else` | `+0.0538` | `+5.51` | `59.4%` |
| `sector19_down_anchor_else` | `+0.0538` | `+5.61` | `58.8%` |
| `anchor_distribution_sideways_sector19_else` | `+0.0515` | `+5.25` | `58.6%` |
| Anchor | `+0.0495` | `+5.01` | `57.5%` |
| Challenger | `+0.0430` | `+4.43` | `56.5%` |

Regime read:

| Regime | Anchor IC | Sector19/router IC | Ket luan |
| --- | ---: | ---: | --- |
| Distribution | `+0.0751` | `+0.0751` | Anchor da tot |
| Sideways | `+0.0577` | `+0.0577` | Anchor da tot |
| Recovery | `+0.0603` | `+0.0603` | It ngay, nhung anchor tot |
| Uptrend | `+0.0377` | `+0.0379` | Gan nhu ngang |
| Downtrend | `-0.0066` | `+0.0243` | Diem yeu ro nhat cua anchor |

Ket luan:

- Edge hien tai la ranking cross-section, khong phai raw return calibration.
- Downtrend la pha can model/expert rieng.
- Phat trien tiep nen tap trung vao rank objective, portfolio objective, hoac regime expert cho downtrend/uptrend.

### 7. Offline rank objective

Report:

`reports/rank_objective_offline/anchor_sector19_rank_objective_20260427_r01`

Validation ranking:

| Candidate | Spearman IC | t-stat | Top-bottom equity | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| `sector19_down_up_anchor_else` | `+0.0539` | `+5.51` | `4.026` | `59.2%` |
| `sector19_down_anchor_else` | `+0.0537` | `+5.59` | `3.847` | `59.1%` |
| `sector19_up_anchor_else` | `+0.0495` | `+4.91` | `3.355` | `58.3%` |
| Anchor | `+0.0493` | `+4.98` | `3.206` | `58.1%` |

Ket luan:

- Router `sector19_down_up_anchor_else` la benchmark rank/portfolio hien tai.
- Anchor van la prediction anchor theo `rel_score`, nhung khong phai best rank/trade candidate.
- Huong tiep theo nen la rank/portfolio sidecar, khong phai train them downtrend expert.

### 8. Train-selected rank router

Report:

`reports/rank_router_train_selected/anchor_sector19_rank_router_20260427_r01`

Rule duoc chon tren train only:

- Global mean-IC weight: `0.4`
- Global top-bottom-equity weight: `0.6`
- Regime mean-IC weights: `{'distribution': 0.55, 'downtrend': 0.8, 'sideways': 0.05}`
- Regime top-bottom-equity weights: `{'distribution': 0.7, 'downtrend': 0.45, 'sideways': 0.35}`

Validation read:

| Candidate | rel_score | IC | Top-bottom equity | Quartile equity | Worst-year equity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sector19_down_up_anchor_else` | `+0.0034` | `+0.0539` | `4.026` | `4.096` | `1.365` |
| `train_rank_regime_ic_weight` | `+0.0037` | `+0.0513` | `3.790` | `4.250` | `1.472` |
| Anchor | `+0.0053` | `+0.0493` | `3.206` | `3.241` | `1.266` |

Ket luan:

- Co cai thien trade-side thuc te: quartile equity va worst-year equity cao hon simple router.
- Khong cai thien IC, nen chua thay `sector19_down_up_anchor_else` lam rank benchmark.
- Nen giu `train_rank_regime_ic_weight` la trade candidate moi, nhung prediction anchor van la `general_sector_full`.

### 9. Stock reliability va stock-bias checks

Reports:

- `reports/stock_reliability_filter/anchor_sector19_stock_reliability_20260427_r01`
- `reports/stock_bias_calibration/anchor_sector19_stock_bias_calibration_20260427_r01`

Ket qua stock reliability filter:

| Candidate rule | Codes | rel_score | IC | Quartile equity | Worst-year equity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_rank_regime_ic_weight__all_stocks` | `28` | `+0.0037` | `+0.0513` | `4.250` | `1.472` |
| `sector19_down_up_anchor_else__all_stocks` | `28` | `+0.0034` | `+0.0539` | `4.096` | `1.365` |
| `anchor__all_stocks` | `28` | `+0.0053` | `+0.0493` | `3.241` | `1.266` |
| Best filtered variant | `21` | `+0.0004` | `+0.0449` | `2.950` | `1.375` |

Ket luan: drop/keep stock theo train per-stock rel_score lam mat breadth va giam IC/equity. Khong dung lam trade filter.

Ket qua stock-bias calibration:

| Candidate rule | rel_score | IC | Quartile equity | Worst-year equity |
| --- | ---: | ---: | ---: | ---: |
| Anchor | `+0.0053` | `+0.0493` | `3.241` | `1.266` |
| `train_rank_regime_ic_weight` | `+0.0037` | `+0.0513` | `4.250` | `1.472` |
| `train_rank_regime_ic_weight__bias_mean_ic_s0p5` | `+0.0046` | `+0.0462` | `3.720` | `1.436` |

Ket luan: bias calibration co the tang raw rel_score nhe, nhung lam giam ranking/trade. Khong dua vao active path luc nay.

## Scripts Moi / Da Cap Nhat

| File | Muc dich |
| --- | --- |
| `experiments/analysis/analyze_prediction_router.py` | Tao candidate router/ensemble tu anchor va challenger |
| `experiments/analysis/analyze_router_weight_grid.py` | Grid weight anchor/challenger, co plot va report gon |
| `experiments/analysis/analyze_train_selected_router.py` | Chon tham so router tren train, evaluate validation de tranh overfit |
| `experiments/analysis/analyze_cross_sectional_ic.py` | Tinh daily cross-sectional IC/t-stat theo candidate, regime, year |
| `experiments/analysis/analyze_regime_performance.py` | Regime labels va regime trade/filter summary |
| `experiments/analysis/analyze_phase_feature_ic.py` | Feature IC theo phase/regime |
| `experiments/analysis/analyze_cycle_phase_report.py` | Chu ky/phase report tu market proxy |
| `experiments/analysis/analyze_downtrend_sidecar.py` | Kiem tra sidecar downtrend tren anchor/router hien tai |
| `experiments/analysis/analyze_rank_objective_offline.py` | Kiem tra Spearman IC, pairwise rank loss, va top-bottom portfolio cho router candidates |
| `experiments/analysis/analyze_train_selected_rank_router.py` | Chon router weights tren train rank objective roi validate |
| `experiments/analysis/analyze_stock_reliability_filter.py` | Kiem tra stock filtering chon tren train-only |
| `experiments/analysis/analyze_stock_bias_calibration.py` | Kiem tra per-stock bias calibration chon tren train-only |
| `experiments/training/run_current_best_signmag_feature_pruning.py` | Runner cho feature, objective, window smoke batches |

## Report Can Doc Khi Quay Lai

| Report | Noi dung |
| --- | --- |
| `docs/current_best_path.md` | Quyet dinh ngan gon hien tai |
| `reports/feature_pruning/broad_signmag_prune_20260424_r04/summary.md` | Sector-generalized follow-up |
| `reports/feature_pruning/broad_signmag_prune_20260425_r09/summary.md` | Phase IC sector follow-up |
| `reports/feature_pruning/broad_signmag_prune_20260426_r02/summary.md` | Objective/window smoke |
| `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01/summary.md` | Weight-grid router |
| `reports/router_train_selected/anchor_sector19_train_selected_20260426_r01/summary.md` | Train-selected router |
| `reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01/summary.md` | Daily cross-sectional IC |
| `reports/rank_objective_offline/anchor_sector19_rank_objective_20260427_r01/summary.md` | Offline rank/portfolio objective check |
| `reports/rank_router_train_selected/anchor_sector19_rank_router_20260427_r01/summary.md` | Train-selected rank router |
| `reports/stock_reliability_filter/anchor_sector19_stock_reliability_20260427_r01/summary.md` | Stock reliability filter check |
| `reports/stock_bias_calibration/anchor_sector19_stock_bias_calibration_20260427_r01/summary.md` | Stock bias calibration check |
| `docs/downtrend_expert_findings.md` | Audit hard-filter, sidecar, va soft-weight downtrend |

## Quyet Dinh Hien Tai

Nen giu:

- `target_mode=return`
- `loss=rel_score`
- `window_size=15`
- `lstm_signmag` lam family chinh
- `general_sector_full` lam prediction anchor
- `phase_ic_sector19` lam trade challenger
- `sector19_down_up_anchor_else` lam router candidate dang xem tiep

Chua nen lam:

- Chot `w_challenger=0.75` vi no dep tren validation weight-grid nhung train-selected router khong xac nhan manh bang.
- Them feature moi neu chua giai thich duoc no tang IC o regime nao.
- Chay out-sample.
- Doi loss/window tiep neu chi dua vao equity validation.

## Huong Tiep Theo

Huong tiep theo nen la mot trong hai nhanh:

1. Train-only regime/router selection:

- Dung rolling IC/regime/volatility tren train de chon anchor vs sector19.
- Muc tieu la vuot `sector19_down_up_anchor_else` ma khong overfit validation.
- Khong train them expert moi tru khi can A/B sach de xac nhan mot gia thuyet cu the.

2. Rank/portfolio objective:

- Vi IC tot hon raw `rel_score`, nen thu objective gan voi cross-sectional ranking.
- Co the bat dau bang analysis/offline loss truoc, chua can sua architecture lon.
- Metric can theo doi: daily IC, top-bottom quartile return, worst-year equity, va validation `rel_score`.

Dieu kien de mo out-sample:

- Model/router duoc chon truoc.
- Khong doi feature/loss/router sau khi xem out-sample.
- Report cuoi phai co train, validation, IC by regime/year, trade equity, drawdown, va histogram errors/relative_score.
