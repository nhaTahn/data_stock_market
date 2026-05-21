# Multi-Market Feature Normalization

Muc tieu: giu cac feature hien co, nhung bien chung thanh cac view dau vao phu hop hon cho LSTM tren du lieu nhieu thi truong.

Voi thi truong \(m\), ma co phieu \(i\), ngay \(t\), feature \(j\):

\[
f_{m,i,t,j}
\]

Input moi co dang:

\[
x_{m,i,t} =
[
x^{roll}_{m,i,t},
x^{csz}_{m,i,t},
x^{rank}_{m,i,t},
x^{market}_{m,t},
x^{calendar}_t
]
\]

Sequence LSTM:

\[
X_{m,i,t} = [x_{m,i,t-L+1}, \ldots, x_{m,i,t}]
\]

voi \(L=15\) theo cau hinh hien tai.

## 1. Rolling per-stock z-score

Dung cho feature ky thuat theo tung ma, vi du `momentum_20`, `volatility_20`, `volume_ratio_20`, `bb_width`, `macd_hist`, `gap_open`.

\[
\mu^{roll}_{m,i,t,j}
=
\frac{1}{w}
\sum_{s=t-w}^{t-1} f_{m,i,s,j}
\]

\[
\sigma^{roll}_{m,i,t,j}
=
\sqrt{
\frac{1}{w-1}
\sum_{s=t-w}^{t-1}
(f_{m,i,s,j}-\mu^{roll}_{m,i,t,j})^2
}
\]

\[
x^{roll}_{m,i,t,j}
=
\frac{
f_{m,i,t,j}-\mu^{roll}_{m,i,t,j}
}{
\sigma^{roll}_{m,i,t,j}+\epsilon
}
\]

Trong code, view nay sinh cot `feature_roll_z`.

## 2. Cross-sectional z-score va rank

Dung cho feature can so sanh giua cac ma trong cung thi truong va cung ngay. Universe:

\[
\mathcal{U}_{m,t}
=
\{i: i \in m \text{ tai ngay } t\}
\]

\[
\mu^{cs}_{m,t,j}
=
\frac{1}{|\mathcal{U}_{m,t}|}
\sum_{i\in \mathcal{U}_{m,t}} f_{m,i,t,j}
\]

\[
\sigma^{cs}_{m,t,j}
=
\sqrt{
\frac{1}{|\mathcal{U}_{m,t}|-1}
\sum_{i\in \mathcal{U}_{m,t}}
(f_{m,i,t,j}-\mu^{cs}_{m,t,j})^2
}
\]

\[
x^{csz}_{m,i,t,j}
=
\frac{
f_{m,i,t,j}-\mu^{cs}_{m,t,j}
}{
\sigma^{cs}_{m,t,j}+\epsilon
}
\]

\[
x^{rank}_{m,i,t,j}
=
\frac{
rank_{i\in \mathcal{U}_{m,t}}(f_{m,i,t,j}) - 1
}{
|\mathcal{U}_{m,t}| - 1
}
\]

Trong code, hai view nay sinh cot `feature_cs_z` va `feature_cs_rank`.

## 3. Market/context feature

Cac feature chung theo ngay nhu `vnindex_return`, `market_leader_return`, `a_d_ratio`, `market_return_20` khong duoc cross-sectional normalize. Chung duoc chuan hoa theo rolling market-level:

\[
x^{market}_{m,t,k}
=
\frac{
c_{m,t,k}-\mu^{market}_{m,t,k}
}{
\sigma^{market}_{m,t,k}+\epsilon
}
\]

Trong code, view nay sinh cot `feature_market_roll_z`.

## 4. Calendar feature

`day_of_week` khong duoc xem nhu bien lien tuc. Thay vao do:

\[
dow^{sin}_{t}
=
\sin(2\pi \frac{dow_t}{5})
\]

\[
dow^{cos}_{t}
=
\cos(2\pi \frac{dow_t}{5})
\]

Trong code, hai view nay sinh cot `day_of_week_sin` va `day_of_week_cos`.

## 5. Cach bat trong train

Mac dinh cac run cu khong doi:

```bash
--feature-normalization-mode none
```

Bat input normalization moi:

```bash
python3 main.py train \
  --feature-normalization-mode multimarket_v1 \
  --feature-normalization-window 60 \
  --feature-normalization-min-periods 20
```

Neu can A/B strict hon, giu rolling window chi dung qua khu den \(t-1\), day la mac dinh. Chi dung `--allow-current-day-feature-roll` khi muon rolling statistic ket thuc tai ngay \(t\).
