# Tom Tat Kien Thuc Toan Hoc Va Huong Toi Uu Mo Hinh

Tai lieu nay dung de chen vao bao cao tong hop cho giang vien, tap trung vao hai cau hoi:

1. De tai dang ap dung nhung kien thuc toan hoc nao.
2. Mo hinh hien tai can toi uu tiep o dau va vi sao.

## 1. Muc tieu mo hinh

De tai huong toi bai toan du bao `next-day return` cho co phieu VN. Nghia la, voi moi co phieu va moi ngay giao dich `t`, mo hinh co gang uoc luong muc sinh loi ngay ke tiep dua tren lich su gia, khoi luong, bien dong, va boi canh thi truong.

Ve mat toan hoc, bai toan nay khong chi la bai toan hoi quy thong thuong. Ket qua nghien cuu hien tai cho thay edge chinh nam o:

- du bao thu tu tuong doi giua cac co phieu trong cung mot ngay;
- danh gia su khac biet giua cac regime thi truong;
- giam do nhay voi noise bang cac metric robust thay vi chi toi uu MSE.

## 2. Cac kien thuc toan hoc da duoc van dung

### 2.1. Bien doi chuoi thoi gian va thong ke mo ta

Phan lon feature dau vao duoc xay tren co so bien doi chuoi thoi gian tai chinh:

- suat sinh loi ngay:

```text
ret_t = A_t / A_(t-1) - 1
```

- dong luong ngan han va trung han:

```text
momentum_5 = A_t / A_(t-5) - 1
momentum_20 = A_t / A_(t-20) - 1
```

- khoang cach so voi xu huong dai han:

```text
ma_200_gap = A_t / MA_200 - 1
```

- do lech chuan cua loi nhuan trong cua so truot:

```text
volatility_20 = STD(ret_(t-19:t))
```

Day la ung dung truc tiep cua:

- thong ke mo ta;
- trung binh truot;
- do lech chuan;
- bien doi ty le va % thay doi;
- phan tich chuoi thoi gian theo cua so truot.

Trong `Feature-Select.pdf`, cac feature nhu `intraday_return`, `gap_open`, `close_position`, `momentum_5`, `momentum_20`, `rolling_max_20_gap`, `ma_200_gap`, `volume_ratio_20`, `volatility_20`, `bb_width` deu duoc xay tren nhung y tuong nay.

### 2.2. Toan hoc trong thiet ke feature ky thuat

Bo feature hien tai khong lay ngau nhien ma duoc xay theo nhom toan hoc ro rang:

- nhom gia va hinh dang nen:
  su dung ty le, chenh lech, vi tri tuong doi trong khung gia ngay;
- nhom trend va momentum:
  su dung trung binh truot, khoang cach toi dinh cu, khoang cach toi duong xu huong;
- nhom thanh khoan va volatility:
  su dung VWAP proxy, ti so khoi luong, do lech chuan, Bollinger band;
- nhom oscillator:
  RSI, MACD histogram, cac bien effort-result va buying/selling pressure;
- nhom context thi truong:
  `a_d_ratio`, `vnindex_return`, `day_of_week`, feature theo nganh va thi truong chung.

Noi cach khac, de tai dang van dung:

- ty le va chuan hoa tuong doi;
- thong ke cua so truot;
- phan tich dao dong va xung luc;
- tong hop tin hieu cross-sectional va market context.

### 2.3. Hoc may va toi uu hoa

Mo hinh chinh hien tai thuoc ho LSTM va `sign-magnitude`.

Ve mat toan hoc, LSTM la mo hinh hoc chuoi du lieu, trong do:

- moi buoc thoi gian duoc anh xa thanh vector dac trung;
- cac cong sigmoid va `tanh` dieu khien cong quen, cong nho, cong cap nhat;
- tham so duoc hoc bang gradient descent va backpropagation through time;
- Adam duoc dung de toi uu ham mat.

Trong family `sign-magnitude`, bai toan duoc tach thanh:

- du bao dau cua return;
- du bao do lon cua return;
- ghep lai thanh:

```text
signed_prediction = sign * magnitude
```

Cach tach nay dua tren mot y tuong toan hoc hop ly: huong bien dong va do lon bien dong khong nhat thiet phai hoc bang cung mot bieu dien.

### 2.4. Thong ke robust va metric danh gia

Mot diem quan trong cua de tai la khong dung duy nhat MSE de danh gia, vi du lieu tai chinh co nhieu noise va ngoai le.

Metric trung tam hien tai la `rel_score`, co dang:

```text
loss(x) = q50(|x|) + 0.5 * q90(|x|)
rel_score = 1 - loss(error) / loss(base)
```

Trong do:

- `q50` la trung vi;
- `q90` la phan vi 90%;
- `error = actual - prediction`;
- `base = actual`.

Day la ung dung cua:

- thong ke thu tu;
- quantile statistics;
- tu duy robust statistics;
- danh gia tuong doi giua sai so va muc bien dong nen cua thi truong.

Uu diem cua cach nay la:

- bot nhay hon voi outlier;
- phan biet ro giua mo hinh that su giam duoc sai so va mo hinh chi “an may” tren vai diem;
- phu hop hon voi du lieu return co phan phoi lech va day duoi day.

### 2.5. Xep hang cross-sectional va thong ke thu hang

Ket qua nghien cuu hien tai chi ra rang edge chinh khong nam o muc du bao return tuyet doi, ma nam o xep hang co phieu trong cung mot ngay.

De danh gia dieu nay, de tai su dung:

- Spearman rank correlation, hay daily cross-sectional IC:

```text
IC_t = Spearman(prediction_t, actual_t)
```

- top-bottom return:

```text
R_top-bottom,t = mean(return_top_quartile) - mean(return_bottom_quartile)
```

- quartile equity;
- hit rate;
- worst-year equity;
- max drawdown.

Day la kien thuc thuoc:

- thong ke thu hang;
- tuong quan phi tham so;
- danh muc long-short;
- phan tich hieu nang theo thoi gian.

Ket qua hien tai cho thay:

- anchor tot ve `rel_score`;
- router theo regime tot hon ve Spearman IC;
- huong toi uu tiep theo nen la rank objective hoac portfolio sidecar.

### 2.6. Kiem dinh tinh on dinh va tranh overfit

De tai khong chi nhin vao metric tong hop ma con tach theo:

- train va validation;
- tung regime thi truong;
- tung nam;
- tung candidate router;
- tung feature set.

Day la ung dung cua:

- train/validation split theo truc thoi gian;
- phan tich do on dinh theo nam;
- thong ke t-stat cho IC;
- kiem tra multiple-candidate de tranh ket luan vo toi va.

Ve mat nghien cuu, day la phan rat quan trong vi mo hinh tai chinh rat de overfit neu chi nhin mot metric dep tren toan tap validation.

## 3. Kien thuc toan hoc duoc ap dung cu the trong phan feature selection

Co the tom tat phan `Feature-Select.pdf` thanh 5 nhom kien thuc chinh:

| Nhom | Kien thuc toan hoc | Vi du |
| --- | --- | --- |
| Price shape | Ty le, vi tri tuong doi, chenh lech gia | `intraday_return`, `gap_open`, `close_position`, `upper_shadow`, `lower_shadow` |
| Trend / momentum | % thay doi, rolling max, moving average | `momentum_5`, `momentum_20`, `rolling_max_20_gap`, `ma_200_gap` |
| Liquidity / volatility | VWAP proxy, MA khoi luong, STD, Bollinger band | `vwap_gap`, `volume_ratio_20`, `volatility_20`, `bb_width` |
| Oscillator / pressure | Chi bao dao dong va luc cung cau | `rsi_14`, `macd_hist`, `buying_pressure`, `selling_pressure` |
| Context / cross-section | Ti le breadth, benchmark, ranking theo thi truong | `a_d_ratio`, `vnindex_return`, `day_of_week`, sector features |

Tu goc nhin hoc thuat, day la su ket hop giua:

- thong ke tai chinh;
- phan tich ky thuat co dinh luong;
- phan tich chuoi thoi gian;
- feature engineering cho machine learning.

## 4. Hien tai mo hinh dang can toi uu o dau

### 4.1. Toi uu muc tieu hoc: tu raw return sang ranking

Ket qua moi nhat cho thay bai toan khong nen duoc xem thuần la hoi quy return.

Ly do:

- model anchor dat `rel_score` tot nhung chua phai la tot nhat ve ranking;
- downtrend la regime model anchor yeu ro nhat;
- router theo regime cai thien Spearman IC ro hon cai thien raw calibration.

Vi vay, huong toi uu hop ly la:

- giu `rel_score` lam prediction anchor;
- them `rank sidecar` hoac portfolio sidecar trong huan luyen;
- danh gia them bang Spearman IC, top-bottom equity, quartile equity.

Noi cach khac, bai toan can duoc toi uu theo hai lop:

1. prediction quality;
2. cross-sectional ordering quality.

### 4.2. Toi uu feature theo tinh portable, khong theo tinh “hop mot giai doan”

Truoc day feature nay bi dat ten va hien thuc qua hep thanh `vingroup_momentum`. Huong dung hon la `market_leader_return`: basket dong cua cac ma anh huong thi truong, duoc chon theo thanh khoan/gia tri giao dich lich su thay vi hard-code rieng cum Vingroup.

Huong toi uu hien tai la:

- uu tien feature co y nghia tong quat hon;
- thay feature hard-code bang feature theo sector breadth, sector rank, alpha so voi nganh;
- cat bo cac feature trung lap ve thong tin.

Do do, feature selection khong chi la “tim feature co metric cao nhat”, ma la bai toan can bang giua:

- suc du bao;
- tinh giai thich duoc;
- kha nang tong quat hoa;
- tinh ben vung qua regime.

### 4.3. Toi uu theo regime thi truong

Mot dong gop toan hoc quan trong cua de tai la nhan ra thi truong khong dong nhat.

Hieu qua cua model thay doi theo:

- uptrend;
- downtrend;
- sideways;
- distribution;
- recovery.

Tu do, bai toan toi uu chuyen thanh:

- khong tim mot bo tham so duy nhat cho moi tinh huong;
- ma tim cach route du bao hoac phoi hop model theo regime.

Day la huong hop ly hon viec train rieng mot downtrend expert, vi thuc nghiem gan day cho thay hard-filter va soft-weighting deu chua du tot.

### 4.4. Toi uu de giam overfitting

Trong bai toan tai chinh, overfitting la rui ro lon nhat. Cac diem can toi uu them:

- khong mo rong qua nhieu bien the loss/window cung luc;
- uu tien train-only selection truoc khi chon router;
- tach ro validation cho model selection va out-of-sample cho xac nhan cuoi;
- theo doi t-stat, so ngay IC duong, worst-year equity thay vi chi nhin mot diem trung binh;
- han che multiple testing khong kiem soat.

Day la noi dung rat quan trong khi bao cao voi giang vien, vi no cho thay de tai khong chi chay mo hinh ma con co tu duy thuc nghiem va kiem dinh.

## 5. Doan tong ket co the chen truc tiep vao bao cao

Co the dung nguyen van doan sau:

> Trong de tai nay, em ap dung tong hop cac kien thuc toan hoc thuoc nhom thong ke, chuoi thoi gian, hoc may va toi uu hoa. O tang feature engineering, em su dung cac phep bien doi ty le, phan tram thay doi, trung binh truot, do lech chuan, Bollinger band, VWAP proxy, RSI, MACD va cac chi so breadth de mo ta dong luc gia, thanh khoan va boi canh thi truong. O tang mo hinh, em su dung LSTM va sign-magnitude decomposition de hoc cau truc chuoi thoi gian, dong thoi toi uu tham so bang gradient-based optimization. O tang danh gia, em khong chi dung cac metric sai so co dien ma uu tien `rel_score`, la mot metric robust dua tren quantile, ben canh daily Spearman IC, top-bottom return va quartile equity de do nang luc xep hang co phieu theo ngay. Ket qua hien tai cho thay huong toi uu tiep theo khong phai la tang do phuc tap kien truc mot cach dai tra, ma la cai thien kha nang ranking cross-sectional, toi uu theo regime thi truong, va kiem soat overfitting thong qua validation theo thoi gian va cac chi so on dinh nhu t-stat, hit rate, worst-year equity va drawdown.

## 6. Phien ban ngan hon de dua vao slide

> De tai van dung cac kien thuc toan hoc chinh gom: thong ke mo ta va chuoi thoi gian de tao feature, hoc may chuoi LSTM de hoc dong luc gia, thong ke robust dua tren quantile de xay dung `rel_score`, va thong ke thu hang nhu Spearman IC de danh gia kha nang xep hang co phieu. Huong toi uu hien tai la giam overfit, uu tien feature portable, va chuyen trong tam tu du bao return tuyet doi sang ranking va regime-aware routing.

## 7. Tien xu ly du lieu va sanity data

Phan nay nen dua vao bao cao de lam ro rang rang truoc khi train model, du lieu da duoc lam sach va kiem tra theo mot quy trinh dinh luong, khong phai dua thang du lieu th raw vao LSTM.

### 7.1. Muc tieu cua pre-processing

Tien xu ly du lieu nham 4 muc tieu:

1. bao dam du lieu co tinh hop le ve mat tai chinh;
2. loai bot nhung dong du lieu co kha nang gay nhieu noise hoac leak;
3. tao feature co y nghia ve mat dinh luong;
4. dua du lieu ve dang sequence phu hop cho LSTM.

### 7.2. Cac buoc clean va sanity check trong quality dataset

Trong pipeline build dataset, repo dang thuc hien cac sanity check sau:

- ep kieu so cho cac cot gia va khoi luong: `open`, `high`, `low`, `close`, `adjust`, `volume_match`, `value_match`;
- danh dau dong thieu cot bat buoc;
- phat hien trung lap theo cap `(code, Date)`;
- phat hien gia am hoac khoi luong am;
- kiem tra tinh hop le cua OHLC:

```text
high >= max(open, close, low)
low <= min(open, close, high)
high >= low
```

- danh dau `has_hard_issue` neu dong du lieu vi pham bat ky dieu kien nao o tren.

Day la buoc rat quan trong, vi neu khong loc cac loi co ban nay thi mo hinh se hoc theo data artifact thay vi hoc theo dong luc gia that.

### 7.3. Loc theo quality o muc ticker

Sau khi clean theo tung dong, repo tong hop quality theo tung ma co phieu. Cac tieu chi chinh gom:

- `coverage_pct`: ti le ngay du lieu cua ma do so voi tong so ngay trong khoang train;
- `days_since_latest`: so ngay cach moc giao dich gan nhat;
- `hard_issue_rows`;
- `imputed_rows`;
- `event_rows`.

Voi thi truong VN, cau hinh hien tai dang dung:

- `min_coverage = 0.95`;
- `recent_active_tolerance_days = 30`;
- `drop_imputed_value_match = true`;
- `drop_neighbors_around_events = true`.

Nghia la, de duoc giu lai trong bo `quality dataset`, mot ma phai:

- co do phu du lieu it nhat 95%;
- van con hoat dong gan hien tai trong nguong 30 ngay;
- khong chua qua nhieu dong loi nghiem trong;
- khong phu thuoc vao cac dong bi imput gia tri khop lenh.

### 7.4. Loc event va tranh du lieu bat thuong

Repo co them mot lop sanity check cho cac bien dong bat thuong theo san giao dich.

Vi du voi VN:

- HOSE co nguong rieng cho `close_return` va `adjust_return`;
- HNX va UPCOM co nguong rieng khac;
- cac dong co bien dong vuot nguong bi danh dau `event_row`.

Sau do:

- dong event bi loai;
- dong `T-1` truoc event cung bi loai neu bat `drop_neighbors_around_events`.

Y nghia hoc thuat cua buoc nay la:

- han che anh huong cua corporate actions, anomaly, split adjustment, va cac phien giao dich rat khac thuong;
- giam kha nang model hoc theo shock hiem gap;
- tranh look-ahead. Repo da ghi ro rang chi buffer `T-1`, khong buffer `T+1`, vi buffer `T+1` se dua thong tin tuong lai vao qua trinh loc du lieu.

### 7.5. Tao feature va target

Sau khi qua vong sanity, du lieu duoc dua qua buoc feature engineering:

- return features;
- price-shape features;
- volume features;
- momentum features;
- volatility features;
- moving-average gap features;
- Bollinger features;
- MACD / RSI;
- price-volume / OBV;
- Wyckoff / VSA;
- feature lich giao dich;
- feature context theo thi truong va theo nganh.

Target du bao duoc tao theo tung ma:

```text
target_next_return = adjust_(t+1) / adjust_t - 1
target_next_3d_return = adjust_(t+3) / adjust_t - 1
target_next_5d_return = adjust_(t+5) / adjust_t - 1
```

Trong nghien cuu hien tai, target chinh la `target_next_return`.

### 7.6. Pre-processing ngay truoc khi vao model

Truoc khi train, repo thuc hien them mot lop tien xu ly cho hoc may:

- tach `train / validation / test` theo truc thoi gian;
- fit `feature scaler` chi tren tap train;
- chuan hoa feature theo z-score:

```text
x_scaled = (x - mean_train) / std_train
```

- tao sequence theo tung ma co phieu voi `window_size`;
- co the bat `instance_zscore` de chuan hoa tung cua so sequence neu can;
- co the dung `target normalizer` cuc bo, vi du chia target cho `volatility_20`, de dua bai toan ve ty le return theo muc bien dong dia phuong;
- target sau do tiep tuc duoc standardize bang `TargetScaler`.

Nghia la, de tai dang ap dung hai lop chuan hoa:

1. chuan hoa feature;
2. chuan hoa target / local target scale.

Ve mat toan hoc, day la cach dua cac bien co don vi va bien do khac nhau ve cung mot khung hoc, giup gradient on dinh hon va giam domination cua mot vai feature co scale qua lon.

## 8. Ly giai LSTM dang dung va cac hien thuc mo hinh trong repo

### 8.1. Tai sao chon LSTM

LSTM phu hop voi bai toan nay vi du lieu co phieu la du lieu chuoi thoi gian, trong do:

- trang thai hien tai phu thuoc vao nhieu ngay truoc do;
- tac dong cua feature khong hoan toan tuyen tinh;
- do tre cua tin hieu co the ngan hoac dai tuy regime;
- can ghi nho co chon loc thay vi nho toan bo lich su.

So voi hoi quy tuyen tinh hoac MLP, LSTM co uu diem:

- mo hinh hoa duoc thu tu thoi gian;
- hoc duoc phu thuoc phi tuyen;
- co co che cong nho / cong quen de loc thong tin.

### 8.2. Cong thuc LSTM co ban

Ve mat ly thuyet, mot LSTM cell gom cac cong:

```text
f_t = sigma(W_f [h_(t-1), x_t] + b_f)
i_t = sigma(W_i [h_(t-1), x_t] + b_i)
g_t = tanh(W_g [h_(t-1), x_t] + b_g)
c_t = f_t * c_(t-1) + i_t * g_t
o_t = sigma(W_o [h_(t-1), x_t] + b_o)
h_t = o_t * tanh(c_t)
```

Trong do:

- `x_t` la vector feature tai thoi diem `t`;
- `h_t` la hidden state;
- `c_t` la memory cell;
- `f_t` quyet dinh quen bao nhieu;
- `i_t` quyet dinh nap bao nhieu thong tin moi;
- `o_t` quyet dinh xuat bao nhieu thong tin ra ngoai.

Y nghia toan hoc cua LSTM la no hoc mot ham phi tuyen co tri nho, phu hop voi du lieu tai chinh co memory ngan-trung han va regime shift.

### 8.3. Plain LSTM trong repo

Family co ban nhat la `plain LSTM`:

- input co dang `(window_size, num_features)`;
- backbone la 1 hoac nhieu tang LSTM xep chong;
- co L2 regularization tren kernel va recurrent weights;
- co dropout giua cac tang recurrent;
- dau ra cuoi cung la `Dense(1)` de du bao return.

Day la baseline hoc sau nhat chinh cua repo.

### 8.4. Sign-magnitude LSTM

Family `sign-magnitude` la family dang duoc tin dung nhat hien tai.

No tach du bao thanh 3 phan:

- `sign_prob`: xac suat return duong;
- `magnitude`: do lon cua bien dong;
- `signed_prediction = sign_centered * magnitude`.

Ve mat ly thuyet, day la mot decomposition hop ly vi:

- huong bien dong va do lon bien dong co the co cau truc thong tin khac nhau;
- bai toan du bao return duoc tach thanh bai toan phan loai + hoi quy.

Repo hien co them `rank sidecar` opt-in cho `signmag`, nghia la:

- van giu `signed_prediction` la output chinh;
- bo sung them `rank_score`;
- toi uu them bang pairwise rank loss theo tung ngay giao dich.

Day la buoc noi huong nghien cuu hien tai voi bang chung rang edge chinh nam o cross-sectional ranking.

### 8.5. Attention family

Repo co `attention LSTM` nhu mot family mo rong:

- sequence tu LSTM duoc dua qua `MultiHeadAttention`;
- co residual connection va layer normalization;
- sau do pooling de lay bieu dien tong hop.

Y nghia:

- thay vi chi lay hidden state cuoi, model co the hoc cach tap trung vao nhung moc thoi gian quan trong hon trong window.

Family nay dang la experimental, khong phai anchor chinh.

### 8.6. Quantile family

Repo co `quantile model` sinh ra:

- `q50`;
- `q90 = q50 + softplus(delta)`.

Loi ich ve mat toan hoc:

- dam bao `q90 >= q50`;
- cho phep mo hinh hoa bat dinh;
- co the dung chenh lech `q90 - q50` nhu proxy uncertainty.

Tuy nhien, family nay hien chi nen xem la sidecar, chua phai huong chinh cua repo.

### 8.7. Event-gated, Signal, PCIE-lite

Repo con co them cac family:

- `event-gated attention`: hoc xac suat xuat hien bien dong lon, roi gate dau ra signed prediction;
- `signal attention`: bien doi sequence thanh patch, tron channel, them temporal attention;
- `pcie-lite`: patching + shared linear ATL + shallow LSTM.

Nhung family nay co gia tri nghien cuu, nhung theo huong hien tai thi:

- plain LSTM va sign-magnitude van la trung tam;
- attention / event / signal / pcie-lite la nhanh phu de so sanh, khong nen mo rong qua nhanh truoc khi giai xong bai toan ranking.

## 9. Hai bao cao histogram can dua vao bao cao

### 9.1. Histogram of errors

Bao cao nay gom tat ca loi du bao tren tat ca co phieu va tat ca ngay trong mot split:

```text
E = prediction - actual
```

Can luu y:

- day la quy uoc dung cho report histogram de doc bias dau duong / dau am cho de;
- trong `rel_score`, repo dung `error = actual - prediction`, nhung vi loss lay tri tuyet doi nen aggregate score khong phu thuoc vao dau cua error;
- histogram `E = prediction - actual` duoc giu lai de phan tich bias.

Tu histogram cua `E`, repo trich:

- `q2 = quantile(E, 0.2)`;
- `q8 = quantile(E, 0.8)`.

Y nghia:

- `q2` la moc ma 20% loi nam ben trai moc nay;
- `q8` la moc ma 80% loi nam ben trai moc nay;
- do rong `q8 - q2` la mot robust error band, it nhay hon so voi min-max.

Neu `E` tap trung quanh 0 va `q8 - q2` hep, mo hinh co sai so on dinh hon.
Neu trung binh cua `E` am dai, mo hinh co xu huong underpredict.
Neu trung binh cua `E` duong dai, mo hinh co xu huong overpredict.

### 9.2. Histogram of relative_score

Do `rel_score` aggregate duoc tinh tu quantile toan tap, repo khong ve histogram truc tiep cua mot scalar `rel_score`, ma ve histogram cua `stabilized local relative_score proxy`.

Y tuong la:

```text
proxy_i = 1 - |actual_i - prediction_i| / max(|actual_i|, proxy_floor)
```

trong do:

```text
proxy_floor = max(base_loss, 1e-4)
base_loss = q50(|actual|) + 0.5 * q90(|actual|)
```

Histogram nay cho biet:

- ty le row co sai so nho hon muc bien dong nen;
- muc do lech trai cua nhung row rat kho du bao;
- su khac biet giua trung binh local proxy va aggregate `rel_score`.

No la mot cong cu chan doan, khong thay the aggregate `rel_score`.

### 9.3. So lieu hien co tu all-stock histogram report

Theo artifact hien co:

- report: `current_vn_all_stock_hist_20260427_r01`;
- pham vi: train va in-sample validation;
- universe: toan bo ma co du bao trong current VN anchor/router frame;
- quy mo hien tai: `28` ma, `36,279` rows train, `18,279` rows validation.
- trong khi do, file `vn_quality_dataset.csv` hien co `93` ma sau quality filtering, va `vn_ticker_quality_summary.csv` dang theo doi `101` ma nguon.

Can ghi chu ro trong bao cao:

> Trong artifact histogram hien tai, "all stocks" co nghia la toan bo ma co prediction trong current VN anchor/router frame, tuong ung 28 ma, khong phai toan bo cleaned VN universe. O thoi diem tong hop nay, cleaned VN quality dataset dang co 93 ma sau quality filtering. Neu can bao cao dung theo nghia "toan bo ma VN hien co trong bo du lieu sach", can tai tao histogram truc tiep tren full run / full cleaned dataset 93 ma.

### 9.4. So lieu validation co the dua thang vao bao cao

| Candidate | rel_score | Error q2 | Error q8 | Error mean | Error std | Proxy > 0 | Rows | Stocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor` | `+0.00534` | `-0.01604` | `+0.01333` | `-0.00080` | `0.02365` | `91.66%` | `18,279` | `28` |
| `sector19_down_up_anchor_else` | `+0.00339` | `-0.01632` | `+0.01308` | `-0.00111` | `0.02361` | `91.84%` | `18,279` | `28` |
| `train_rank_regime_ic_weight` | `+0.00367` | `-0.01632` | `+0.01301` | `-0.00111` | `0.02362` | `91.75%` | `18,279` | `28` |

Doc nhanh bang nay:

- anchor van tot nhat ve `rel_score`;
- hai candidate router/rank co error band gan anchor nhung co bias am hon mot chut;
- histogram va error band xac nhan bai toan hien tai khong nam o viec cat giam error tuyet doi qua manh, ma o viec sap xep thu tu co phieu tot hon theo ngay.

## 10. Doan van co the chen truc tiep vao bao cao cho 3 phan moi

### 10.1. Phan pre-processing va sanity data

> Truoc khi huan luyen mo hinh, du lieu thi truong duoc dua qua mot quy trinh tien xu ly va kiem tra chat luong nhieu lop. O muc tung dong du lieu, he thong kiem tra gia tri thieu, trung lap theo `(code, Date)`, gia am, khoi luong am, va tinh hop le cua OHLC. O muc tung ma co phieu, he thong danh gia do phu du lieu, do gan cua ngay giao dich cuoi, so dong bi imput, va so dong co bien dong bat thuong. Voi thi truong VN, bo du lieu sach hien tai giu cac ma co do phu toi thieu 95%, con hoat dong trong nguong 30 ngay, va loai bo cac dong co anomaly theo nguong bien dong rieng cua tung san giao dich. Cach lam nay giup giam nhieu, tranh hoc theo data artifact, va tao mot bo du lieu co tinh on dinh cao hon truoc khi dua vao mo hinh sequence.

### 10.2. Phan ly giai LSTM va cac hien thuc mo hinh

> Mo hinh chinh cua de tai la LSTM, vi bai toan du bao gia co phieu la bai toan chuoi thoi gian, trong do thong tin qua khu co anh huong phi tuyen den du bao tuong lai. Ve ly thuyet, LSTM su dung cac cong quen, cong nho va cong xuat de dieu tiet dong thong tin qua thoi gian, nham giu lai nhung tin hieu quan trong va loai bo nhieu khong can thiet. Trong repo hien tai, family plain LSTM dong vai tro baseline hoc sau, con family sign-magnitude la hien thuc quan trong nhat do tach bai toan thanh du bao dau va du bao do lon bien dong. Ben canh do, repo con co cac family attention, quantile, event-gated, signal va pcie-lite de phuc vu nghien cuu mo rong, tuy nhien huong chinh hien tai van uu tien plain LSTM va sign-magnitude de tranh tang do phuc tap qua som.

### 10.3. Phan histogram of errors va relative_score

> De phan tich sau hon hanh vi cua mo hinh, de tai su dung hai bao cao histogram tren toan bo co phieu va toan bo ngay giao dich trong tung split. Bao cao thu nhat la histogram cua sai so `E = prediction - actual`, tu do trich `q2` va `q8` la cac phan vi 20% va 80% cua phan phoi sai so, dung de mo ta robust error band thay cho min-max. Bao cao thu hai la histogram cua `stabilized local relative_score proxy`, dung de quan sat phan bo chat luong du bao tren tung row trong khi van nhat quan voi metric tong hop `rel_score`. Can ghi ro rang rang artifact histogram hien tai moi phan anh 28 ma trong current prediction frame; neu viet theo nghia full cleaned VN universe thi moc dung hien tai la 93 ma sau quality filtering va can tai tao lai histogram tren tap day. Ve y nghia thuc nghiem, cac histogram nay cho thay model anchor hien tai van tot nhat ve `rel_score`, nhung loi the cai thien tiep theo co ve den tu kha nang xep hang co phieu theo ngay va theo regime, hon la chi giam sai so hoi quy thuan tuy.
