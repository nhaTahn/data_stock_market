# Toi Uu Feature Va Danh Gia OOD 2026-04-28

Tai lieu nay tong hop 3 cau hoi:

1. Bo chi bao nao hien tai la phu hop nhat cho anchor model.
2. Huong cai thien nao da thu tren model tot nhat hien tai va ket qua ra sao.
3. Model hien tai da san sang dem sang JP, KR, US de "thuc chien" hay chua.

## 1. Ke hoach toi uu nen giu

Thu tu toi uu hop ly hien tai:

1. khoa bo feature anchor truoc, khong doi lung tung window/loss;
2. chi thu thay doi nho tren model anchor neu co bang chung toan hoc ro rang;
3. danh gia moi thay doi bang 3 lop:
   - prediction quality: `rel_score`, `error_q2`, `error_q8`, directional accuracy;
   - ranking quality: mean daily Spearman IC, IC t-stat, positive IC days;
   - trade proxy: quartile equity, quartile hit rate, max drawdown;
4. sau khi model con tot tren VN validation moi dem di OOD JP/KR/US;
5. neu OOD van kem thi khong merge vao "current best".

## 2. Bo chi bao phu hop nhat hien tai

Ket luan hien tai van giu nguyen:

- feature set dung nhat la `general_sector_full`;
- window dung nhat la `15`;
- objective anchor dung nhat la `rel_score`.

### 2.1. Bang ket qua chinh tu feature pruning

Nguon:

- `reports/feature_pruning/broad_signmag_prune_20260424_r04`
- `reports/feature_pruning/broad_signmag_prune_20260424_r05`
- `reports/feature_pruning/broad_signmag_prune_20260426_r02`

| Case | Val rel_score | Val quartile equity | Doc nhanh |
| --- | ---: | ---: | --- |
| `general_sector_full` | `+0.00534` | `2.477` | Tot nhat tong the |
| `general_sector_breadth` | `+0.00485` | `1.809` | Chi giu breadth la chua du |
| `general_sector_momentum_relative` | `+0.00273` | `1.514` | Chi giu momentum tuong doi la chua du |
| `general_sector_rank_return_alpha` | `+0.00215` | `1.558` | Rank + return/alpha thieu breadth va momentum gap |
| `general_sector_rank_alpha` | `+0.00217` | `0.808` | Alpha don le chua du |
| `general_sector_rank` | `+0.00157` | `1.859` | Rank sector don le chua du |

### 2.2. Giai thich theo huong toan hoc

`general_sector_full` thang vi no ket hop 3 lop thong tin:

- thong tin stock-level:
  candle state, delta, volatility, momentum, MACD;
- thong tin market-level:
  `vnindex_return`, `a_d_ratio`;
- thong tin sector-level:
  `sector_momentum_rank`, `sector_momentum_20`, `relative_sector_momentum_20`, `sector_return`, `alpha_sector`, `sector_positive_ratio`, `sector_ad_ratio`.

Ve mat toan hoc:

- `sector_return` va `alpha_sector` tach thanh phan chung theo nganh va phan du idiosyncratic cua tung co phieu;
- `relative_sector_momentum_20` do do lech dong luong cua co phieu so voi nganh, tuc la mot residual momentum;
- `sector_positive_ratio` va `sector_ad_ratio` do breadth trong nganh, giup phan biet mot tin hieu la "ro rong" hay chi la mot diem le;
- `sector_momentum_rank` cung cap thu tu ordinal cua nganh, co ich cho bai toan rank cross-sectional.

Noi cach khac, `general_sector_full` giam duoc variance cua bai toan bang cach:

- tach stock effect;
- tach sector effect;
- tach market effect;
- roi moi de LSTM hoc phan dong luc con lai.

### 2.3. Ablation xac nhan feature nao thuc su quan trong

| Ablation tu `general_sector_full` | Val rel_score | Val quartile equity | Y nghia |
| --- | ---: | ---: | --- |
| Anchor `general_sector_full` | `+0.00534` | `2.477` | Moc chuan |
| Bo `sector_rank` | `+0.00301` | `1.615` | Thu tu ordinal cua nganh rat quan trong |
| Bo `sector_breadth` | `+0.00259` | `1.157` | Breadth giam manh trade quality |
| Bo `sector_return_alpha` | `+0.00116` | `1.676` | Residual so voi nganh rat co gia tri |
| Bo `sector_momentum_gap` | `-0.00042` | `1.434` | Momentum gap la thanh phan "khong the cat" |
| Bo `sector_rank_pct` | `+0.00240` | `2.440` | Rank pct co ve phan nao trung lap, it quan trong hon rank ordinal |

Ket luan:

- feature quan trong nhat de giu lai la `sector_momentum_gap`, `sector_rank`, `sector_breadth`, `sector_return`, `alpha_sector`;
- `sector_momentum_rank_pct` la ung vien yeu nhat trong nhom sector-full, co the xem nhu mot feature phu hoac muc tieu don gian hoa sau nay.

### 2.4. Window va objective

| Case | Val rel_score | Val quartile equity | Ket luan |
| --- | ---: | ---: | --- |
| Anchor `w15 + rel_score` | `+0.00534` | `2.477` | Giu lam chuan |
| `w20 + rel_score` | `+0.00323` | `1.356` | Kem hon |
| `w10 + rel_score` | `+0.00146` | `1.283` | Kem hon |
| `w15 + rel_score_weighted` | `+0.00276` | `1.708` | Kem hon |
| `w15 + rel_score_sharp` | `+0.00038` | `2.637` | Chi la trade clue, khong du lam anchor |

Ket luan:

- khong doi window khoi `15`;
- khong doi objective anchor khoi `rel_score`;
- `rel_score_sharp` chi nen coi la dau moi trade-side, khong phai loss chinh.

## 3. Thu cai thien tren model tot nhat hien tai

Model anchor dang dung:

- run: `broad_signmag_prune_general_sector_full_20260424_r04`
- model: `lstm_signmag_seed_42`

Baseline VN validation:

| Model | Val rel_score | Dir acc | Mean IC | IC t-stat | Positive IC days | Quartile equity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor `lstm_signmag_seed_42` | `+0.00534` | `48.85%` | `+0.05173` | `+4.99` | `58.57%` | `2.407` |

### 3.1. Thu rank sidecar

Da thu 2 probe nho tren cung feature set anchor:

- `rank_loss_weight = 0.05`
- `rank_loss_weight = 0.01`

Tat ca deu dung seed `42`, train tren CPU wrapper de tranh loi dynamic batch shape cua TensorFlow Metal.

| Probe | Epoch log | Val rel_score | Dir acc | Mean IC | IC t-stat | Positive IC days | Quartile equity | Ket luan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Anchor | - | `+0.00534` | `48.85%` | `+0.05173` | `+4.99` | `58.57%` | `2.407` | Moc chuan |
| `rank=0.05` | `9` | `+0.00142` | `48.15%` | `+0.03601` | `+3.46` | `57.51%` | `2.190` | Kem hon anchor o moi truc |
| `rank=0.01` | `14` | `-0.00111` | `48.76%` | `+0.04554` | `+4.53` | `58.12%` | `1.489` | IC van kem anchor, rel_score am |

### 3.2. Giai thich toan hoc

Rank sidecar hien tai chua nen merge vao anchor vi:

- no them mot rang buoc pairwise tren thu tu cross-sectional;
- rang buoc nay canh tranh truc tiep voi head `signed_prediction` dang duoc toi uu cho `rel_score`;
- khi weight cua pairwise loss tang, representation chung bi keo theo thu tu tuong doi nhieu hon la calibration do lon return;
- ket qua la magnitude calibration xau di nhanh hon phan rank edge thu them.

Noi ngan gon:

- objective moi lam tang ap luc "sap thu tu";
- nhung bai toan anchor hien tai van can giu "do lon hop ly";
- voi 2 probe vua chay, chi phi mat calibration lon hon loi ich rank.

Vi vay:

- khong dua `rank sidecar` hien tai vao `current best`;
- neu muon quay lai huong nay, can sua engineering cua date-grouped batch va thu objective sidecar mem hon, hoac dung mot router sau model thay vi ep vao representation chung qua som.

### 3.3. Ghi chu engineering

Co 2 canh bao ky thuat da lo ra:

- TensorFlow Metal bi loi dynamic batch shape voi date-grouped rank batches;
- khi chay wrapper tat GPU, Keras co warning `input ran out of data`, nhung run van hoan tat va sinh du metrics.

Do do, rank-sidecar hien tai moi dat muc:

- proof-of-concept khong dat;
- chua dat muc production experiment.

## 4. Model hien tai co san sang dem sang JP, KR, US chua

Danh gia OOD da chay tren anchor hien tai:

- script: `experiments/analysis/evaluate_run_ood_readiness.py`
- run nguon: `broad_signmag_prune_general_sector_full_20260424_r04`
- model: `lstm_signmag_seed_42`

Luu y cuc ky quan trong:

- stock identity cua model duoc hoc tren 28 ma VN;
- JP/KR/US deu co `known_code_share = 0%`;
- raw metadata local cua JP/KR/US hien khong co sector map, nen trong test OOD toan bo sector bi collapse ve `Unknown`;
- vi vay sector-aware edge cua anchor khong the duoc chuyen sang day du duoc.

### 4.1. Bang OOD

| Market | Requested | Accepted | Test rows | rel_score | Dir acc | Mean IC | IC t-stat | Quartile equity | Quartile hit rate | Max DD | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `JP50` | `50` | `26` | `21,366` | `-0.00175` | `49.93%` | `+0.02181` | `+2.45` | `1.222` | `53.34%` | `-17.90%` | Co chut rank transfer, calibration van kem |
| `KR50` | `50` | `44` | `35,968` | `-0.00110` | `48.89%` | `+0.03637` | `+5.10` | `1.433` | `53.90%` | `-27.50%` | Rank transfer ro hon JP, nhung DD cao va rel_score van am |
| `US100` | `100` | `89` | `75,017` | `-0.00131` | `51.06%` | `+0.00524` | `+0.76` | `1.111` | `48.28%` | `-13.74%` | Gan nhu khong co rank edge on dinh |

### 4.2. Giai thich

Doc theo toan hoc:

- `rel_score < 0` tren ca 3 thi truong nghia la sai so robust cua model con lon hon moc base;
- `IC > 0` o JP/KR cho thay mot phan ranking signal co transfer;
- nhung signal do chua du manh de chuyen thanh mot model "san sang thuc chien" vi calibration van am va drawdown van cao;
- US yeu nhat, cho thay signal hoc tren VN khong tu dong tong quat sang mot cross-section co microstructure khac.

Doc theo cau truc feature:

- anchor hien tai an nhieu vao sector context va stock identity;
- OOD thi ca 2 lop nay deu bi gay:
  - stock identity = toan bo unseen;
  - sector = `Unknown` cho 100% raw local universe.

Vi vay, ket qua OOD hien tai khong chi noi rang "model chua san sang", ma con noi ro ly do:

- bai toan transfer hoc hien tai dang thieu metadata co cau;
- model chua du portable ve bieu dien.

## 5. Ket luan cuoi

Ket luan ngan:

1. `general_sector_full + window 15 + rel_score` van la anchor tot nhat hien tai.
2. Khong merge `rank sidecar` vao anchor o thoi diem nay.
3. Chua san sang dem model nay di thuc chien JP/KR/US.

Neu xep muc uu tien tiep theo:

1. Giu anchor hien tai lam benchmark.
2. Neu muon portable hon, can tao mot portable branch moi:
   - bo hoac lam mem stock identity;
   - cap sector map that cho JP/KR/US;
   - so sanh lai giua feature set sector-full va mot feature set market-only khong can sector metadata.
3. Chi khi OOD dat toi thieu:
   - `rel_score >= 0`,
   - IC duong on dinh,
   - quartile equity > 1 mot cach ro,
   - drawdown chap nhan duoc,
   thi moi nen noi den "ready for live".

## 6. Hanh dong tiep theo de nghi

De co buoc tien thuc su, nen lam theo thu tu nay:

1. Tao `portable_no_identity` benchmark tren VN:
   - cung feature set anchor,
   - tat stock identity,
   - giu `rel_score`.
2. Tao `portable_no_sector_context` benchmark tren VN:
   - cat `sector_*` va `alpha_sector`,
   - giu market + stock-only features,
   - xem mat bao nhieu edge.
3. Neu 1 trong 2 benchmark tren van giu duoc phan lon edge VN, dem no di OOD lai JP/KR/US.
4. Sau do moi tinh den fine-tune rieng theo thi truong.

Cho den luc chay xong 4 buoc tren, cau tra loi nghiem tuc nhat la:

> Mo hinh hien tai chua san sang de dem thuc chien qua JP, KR, US. No van la mot VN-first model co kha nang transfer mot phan ve ranking, dac biet o KR, nhung chua dat muc portable va calibration du de coi la san pham giao dich da san sang.
