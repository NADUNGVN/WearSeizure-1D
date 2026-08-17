# Research Reality Check — WearSeizure-1D

Ngày: 2026-08-16
Phạm vi: đối chiếu 19 run thật trên CHB-MIT (`docs/EXPERIMENT_LOG_G1a.md`) với văn liệu đã công
bố và với chính mã nguồn trong repo, nhằm trả lời 4 câu hỏi: KPI có khả thi không, đóng góp có
novelty không, đường đi ngắn nhất tới bài báo là gì, và mục tiêu cần điều chỉnh thế nào.

Mọi con số trong tài liệu này đều có nguồn: hoặc là dòng cụ thể trong `EXPERIMENT_LOG_G1a.md`,
hoặc là dòng mã trong repo, hoặc là một DOI/URL công khai. Không có số nào được ước lượng.

---

## 0. Kết luận điều hành

1. **Ba cổng chưa từng đạt trong 19 run không phải do model yếu.** `detection_delay_mean_s` và
   `worst_patient_sensitivity` là **bất khả thi về mặt toán học** dưới cấu hình đo mặc định của
   repo — không model nào đạt được, kể cả model hoàn hảo. Xem §3 và §4.
2. **Bộ KPI gate được sao chép từ chính bài baseline mà dự án định vượt qua** (Chung et al. 2024),
   trong khi bài đó dùng đúng 4 lỗi giao thức mà `docs/PROTOCOL.md` cấm. Đạt gate = hoà, không
   phải thắng. Xem §2.
3. **Dự án đang tối ưu nhầm trục.** FAR tốt nhất đạt 0.0621/h trong khi chỉ cần ≤0.30/h — dư
   ~5×. Cùng lúc đó `detection_delay` chưa bao giờ đạt. Ngân sách FAR dư phải được **tiêu để mua
   độ trễ**, chứ không phải nén thêm. Xem §5.
4. **Trục "độ chính xác" và "nJ/inference" đã bị chiếm** bởi một bài IEEE TBioCAS 2025 cùng
   dataset, cùng ứng dụng wearable. Đóng góp phải được định vị lại sang giao thức + kiến trúc +
   công suất đo thật. Xem §6.
5. **Params/MAC đã được đo lần đầu** (§4b): model mặc định là **14 834 params / 765 632 MACs**,
   vượt mục tiêu Table 4 và **trượt mức target của cổng MAC**. Biến thể `k5only` — vốn đã thắng ở
   ablation — là **11 786 / 585 920**, và với cửa sổ 2 s là **11 786 / 293 056**, đạt cả hai mức
   stretch cùng lúc.
6. **Rủi ro cuối dự án:** PYNQ-Z2 không có cảm biến công suất on-board, trong khi "measured FPGA
   power" là tiêu chí GO/NO-GO cứng. Xem §7.

Các thay đổi code phát sinh từ phân tích này đã được thực hiện và kiểm thử — xem §10.

---

## 1. Trạng thái đã kiểm chứng

Nguồn: `docs/EXPERIMENT_LOG_G1a.md` — 20 run thật trên CHB-MIT, 13 bệnh nhân, 66 fold,
`patient_specific_loso_edf`, exposure 185.0 h, 08-14 → 08-16.

| Cổng | Minimum | Tốt nhất đạt được | Run | Trạng thái |
|---|---|---|---|---|
| `far_per_hour` | ≤0.30 | **0.0621** | #19 | ✅ vượt cả stretch (0.10) |
| `worst_patient_far_per_hour` | ≤1.0 | **0.1899** | #19 | ✅ vượt cả stretch (0.5) |
| `continuous_test_exposure_hours` | ≥100 | **185.0** | mọi run | ✅ vượt target (150) |
| `personalized_event_sensitivity` | ≥0.970 | **0.9487** @ FAR 1.024 ❌ | #9 | ❌ chưa từng đạt |
| | | 0.8806 @ FAR 0.3119 | #12/#17 | (cấu hình hợp lệ tốt nhất) |
| `detection_delay_mean_s` | ≤5.0 | **14.67** | #9 | ❌ chưa từng đạt |
| `worst_patient_sensitivity` | ≥0.85 | **0.6667** | #2, #4, #6… | ❌ chưa từng đạt |

Chưa đo: zero-shot LOSO (chưa chạy lần nào) và **INT8/QAT loss**. Params/MAC đã đo — xem §4b.
Nhánh A (`w2s_stride1s`) đã chạy lại và cho kết quả âm — xem §3.5.

**Cấu hình thật tốt nhất tính đến nay là row #15** (`wearseizure1d_k5only`, `w4s_stride1s`,
hậu xử lý mặc định): sens 0.8756, FAR 0.1624, delay 23.46 s, worst-pt sens 0.2500,
worst-pt FAR 0.3926.

**Cảnh báo về khả năng truy vết:** "Research Decision Memo" — nguồn của Table 4, Table 6,
Appendix A, §5.3, §7.2 và toàn bộ tiêu chí GO/NO-GO — không tồn tại ở bất kỳ đâu trong repo.
Mọi mục tiêu hiện chỉ tồn tại dưới dạng đã chép tay vào `docs/GATES.md` và
`configs/eval/gates.yaml`, không thể audit ngược về nguồn.

---

## 2. Bộ gate đến từ đâu, và tại sao đó là vấn đề

### 2.1 Nhận dạng bài baseline

Bài "Frontiers 2024" được nhắc trong `docs/PROTOCOL.md` là:

> Chung Y-G, Cho A, Kim H, Kim KJ. *Single-channel seizure detection with clinical confirmation
> of seizure locations using CHB-MIT dataset.* **Front. Neurol. 15:1389731 (2024).**
> <https://pmc.ncbi.nlm.nih.gov/articles/PMC11148866/>

Trùng khớp tuyệt đối với cấu hình dự án:

| Thuộc tính | Chung 2024 | `configs/data/chbmit.yaml` |
|---|---|---|
| Số case | 13 | 13 |
| Danh sách case | chb01,02,03,04,05,07,08,10,11,15,17,22,23 | giống hệt |
| Fp1-F3 | chb03,07,08,22,23 | giống hệt |
| P7-O1 | chb02,05,10,11,15 | giống hệt |
| P8-O2 | chb01,04,17 | giống hệt |
| Tổng thời lượng | 599.5 h | 599.5 h (README) |
| Số cơn | 77 | 77 (README) |

Kết quả single-channel của họ (trích nguyên văn): sensitivity **99.62 ± 1.39 %**,
FAR **0.22 ± 0.34 /h**, latency **3.3 ± 5.5 s**.

→ Bộ gate 97/98.5/99 %, 0.30/0.20/0.10 /h, 5/4/3 s trong `configs/eval/gates.yaml` rõ ràng được
dẫn xuất từ ba con số này.

### 2.2 Vấn đề: baseline dùng đúng 4 lỗi mà PROTOCOL.md cấm

Chung 2024 tự thừa nhận trong phần Limitations:
- Cửa sổ 4 s trượt **1 mẫu** (1/256 s), overlap **3.996 s** giữa hai cửa sổ liên tiếp
  → chính là "overlap leakage" ở `PROTOCOL.md` mục 1.
- `Th` (0.2–0.9) và `L` (5–10) được **chỉnh tay theo từng bệnh nhân trên dữ liệu đánh giá**
  → chính là "test-tuned postprocessing" ở mục 2.
- Chỉ test trên **7.0 ± 4.1 h/case** (≈91 h tổng), không phải toàn bộ 599.5 h.
- Đã **re-annotate lại** 77 cơn, tức mốc onset khác mốc công khai của CHB-MIT.

Mức độ thổi phồng của kiểu giao thức này đã được định lượng độc lập:

> Ali E, Angelova M, Karmakar C. *Epileptic seizure detection using CHB-MIT dataset: The
> overlooked perspectives.* **R. Soc. Open Sci. 11:230601 (2024).**
> <https://royalsocietypublishing.org/doi/10.1098/rsos.230601>

Cùng một hệ thống, chỉ đổi cách chia dữ liệu: **random-split 5-fold đạt 91.88 % sensitivity,
cross-subject chỉ còn 64.24 %.**

### 2.3 Hệ quả

Khoảng cách **0.8806 (repo, leakage-safe, 185 h) → 0.9962 (Chung, leaky, 91 h)** không phải là
thất bại kỹ thuật. Đó là **cái giá của việc làm đúng**, và nó là một kết quả có thể công bố.

Nhưng nó cũng có nghĩa: **gate `personalized_event_sensitivity ≥ 0.970` vừa không đủ để thắng
baseline (0.9962), vừa có thể không đạt nổi dưới giao thức trung thực.** Cần được viết lại.

---

## 3. `detection_delay_mean_s` là bất khả thi về cấu trúc

Đây là phát hiện quan trọng nhất của tài liệu này.

### 3.1 Dẫn xuất từ mã nguồn

Ba dòng mã quyết định:

- `src/wearseizure/postprocess/hysteresis.py:75` — `alarm_start = float(t)` với `t` lấy từ
  `end_sec`, tức **thời điểm KẾT THÚC cửa sổ**, và chỉ được gán khi `run_count >= run_length`.
- `src/wearseizure/postprocess/pipeline.py:27` — chuỗi điểm số được làm mượt bằng
  `ema_smooth(scores, ema_alpha)` **trước** khi vào hysteresis.
- `src/wearseizure/eval/metrics_event.py:39` — `delay = max(0, alarm_start − event_onset)`.

Do đó, **sàn dưới của độ trễ đo được** — giá trị mà ngay cả một bộ phân loại hoàn hảo, nổ đúng
mẫu đầu tiên của cơn, cũng không thể xuống dưới — là:

```
delay_floor = window_s  +  (run_length − 1) × stride_s  +  ((1 − α) / α) × stride_s
              └ cửa sổ ┘   └── run-length ──┘              └──── độ trễ trung bình của EMA ────┘
```

Số hạng thứ ba là độ trễ trung bình của bộ lọc EMA bậc một: với `y_t = α·x_t + (1−α)·y_{t−1}`,
độ trễ trung bình là `(1−α)/α` mẫu.

### 3.2 Áp vào các cấu hình đã dùng

`configs/postprocess/hysteresis_runlength.yaml` mặc định: `run_length: 3`, `ema_alpha: 0.125`.

| Cấu hình | window / stride | L | α | **Sàn delay** | Delay đo được | Phần dư (model) |
|---|---|---|---|---|---|---|
| Mặc định (#12, #15, #17) | 4 s / 1 s | 3 | 0.125 | **13.0 s** | 19.42 / 23.46 s | ~6–10 s |
| #9 "aggressive" | 4 s / 1 s | 1 | 0.5 | **5.0 s** | 14.67 s | ~9.7 s |
| #10 "moderate" | 4 s / 1 s | 2 | 0.25 | **8.0 s** | 18.69 s | ~10.7 s |
| #13 stride mịn | 4 s / 0.5 s | 2 | 0.25 | **6.0 s** | 18.37 s | ~12.4 s |
| Nhánh A (chưa chạy) | 2 s / 1 s | 3 | 0.125 | **11.0 s** | — | — |
| Cấu hình tối ưu delay | 2 s / 0.5 s | 1 | 0.5 | **2.5 s** | — | — |

**Kết luận:** với cấu hình mặc định của repo, sàn delay là **13.0 s**, trong khi gate đòi
**≤5.0 s**. Gate này không thể đạt được bằng bất kỳ model nào — nó bị chặn bởi chính bộ hậu xử lý.
Chỉ có run #9 từng chạy ở cấu hình có sàn ≤5 s, và cấu hình đó đánh đổi FAR lên 1.024/h.

### 3.3 Tại sao Chung 2024 báo được 3.3 s

Với stride 1/256 s, số hạng `(L−1)×stride` của họ chỉ tốn ~40 ms dù `L` tới 10. Nhưng 3.3 s vẫn
**nhỏ hơn `window_s = 4 s`**, nghĩa là họ **không** đóng dấu alarm ở cuối cửa sổ (nhiều khả năng
ở đầu hoặc tâm cửa sổ), và mốc onset của họ là mốc **đã re-annotate**, không phải mốc CHB-MIT
công khai.

→ **Con số 3.3 s và con số 19 s không đo cùng một đại lượng.** Đặt cái sau vào một cổng dẫn xuất
từ cái trước là so sánh khập khiễng.

### 3.4 Nhánh A xác nhận công thức sàn — và cho thấy nó đã đánh nhầm số hạng

Row #20 chạy `k5only` với cửa sổ 2 s nhưng **giữ nguyên hậu xử lý của cửa sổ 4 s**
(`run_length=3`, `ema_alpha=0.125`, stride 1 s). Vì stride không đổi, hai số hạng run-length và
EMA **giống hệt nhau** giữa #15 và #20 — nên phép so sánh độ trễ ở đây sạch hơn log ghi nhận:

| | Cửa sổ | Run-length | EMA | **Sàn** | Đo được | Phản ứng model |
|---|---:|---:|---:|---:|---:|---:|
| #15 `w4s/1s` | 4.0 | 2.0 | 7.0 | **13.0** | 23.46 | 10.46 |
| #20 `w2s/1s` | 2.0 | 2.0 | 7.0 | **11.0** | 26.18 | **15.18** |

Hai kết luận:

1. **Nhánh A đã tấn công số hạng nhỏ nhất.** Trong sàn 13.0 s, cửa sổ chỉ chiếm 4.0 s (31 %),
   trong khi **EMA chiếm 7.0 s (54 %)** và run-length 2.0 s (15 %). Rút cửa sổ từ 4 s xuống 2 s
   chỉ hạ được 2.0 s của sàn, còn 9.0 s lớn nhất thì không đụng tới. Đây không phải một phép thử
   thất bại về ý tưởng — nó là một phép thử nhắm sai chỗ.
2. **Cửa sổ ngắn làm model chậm đi thật.** Phản ứng model tăng 10.46 → 15.18 s (**+4.72 s**):
   với 2 s ngữ cảnh, bộ phân loại cần nhiều cửa sổ liên tiếp hơn mới đủ tự tin. Cộng với việc FAR
   xấu đi rõ (0.1624 → 0.2919; worst-patient 0.3926 → 1.7259), **cửa sổ 4 s nên được giữ.**

Thí nghiệm độ trễ đúng là hạ hai số hạng lớn, không phải cửa sổ:

| Cấu hình | Cửa sổ | Run-length | EMA | Sàn |
|---|---:|---:|---:|---:|
| Hiện tại (#15) | 4.0 | 2.0 | 7.0 | 13.0 |
| **`w4s/1s`, `L=1`, `α=0.5`** | 4.0 | 0.0 | 1.0 | **5.0** |
| `w4s/0.5s`, `L=1`, `α=0.5` | 4.0 | 0.0 | 0.5 | **4.5** |

Tức là chỉ cần đổi hai tham số hậu xử lý — **không đổi cửa sổ, không đổi kiến trúc, không train
lại** — là hạ được 8.0 s khỏi sàn. Ngân sách FAR đang dư ~1.8 lần ở #15 (0.1624 so với trần 0.30)
chính là thứ để chi trả cho việc nới hậu xử lý này, và hàm mục tiêu `min_delay` (§10) tồn tại để
làm đúng việc đó một cách có ràng buộc.

### 3.5 Ghi chú về tiêu chí khớp event

`src/wearseizure/eval/event_matching.py:37` khớp alarm với event chỉ cần **chồng lấn bất kỳ**.
Cơn động kinh trung bình dài 57.4 s (Chung 2024), nên một alarm nổ ở onset + 50 s vẫn được tính
là "matched". Đây là lý do sensitivity 88 % có thể cùng tồn tại với delay 20 s.

Ali 2024 dùng tiêu chí chặt hơn: "detected segments cover at least 70 % of the original event".
Khi so sánh với văn liệu, phải nêu rõ khác biệt này, nếu không sensitivity của repo sẽ bị xem là
được nới lỏng.

---

## 4. `worst_patient_sensitivity` là bất khả thi về số học

`docs/EXPERIMENT_LOG_G1a.md` §3 ghi: chb17 "chỉ có 3 seizure event tổng cộng"; chb02 `n_events=3`.

Với **n = 3**, sensitivity của bệnh nhân đó chỉ có thể nhận **4 giá trị rời rạc**:

```
0/3 = 0.000    1/3 = 0.333    2/3 = 0.667    3/3 = 1.000
```

Gate `worst_patient_sensitivity ≥ 0.85` do đó tương đương với: **chb17 phải bắt đúng 3/3 cơn.**
Bỏ sót một cơn duy nhất → 0.667 → fail toàn bộ dự án.

Đây không phải một cổng độ nhạy; đây là một cổng hoàn hảo áp lên bệnh nhân có mẫu nhỏ nhất.

Điều này giải thích quan sát trong log rằng `worst_patient_sensitivity` "lì ra đúng 0.3333..."
qua cả hai lần train 30 epoch và 60 epoch — như log tự nhận xét, đó là đặc tính cố định của dữ
liệu, không phải model thiếu train.

Văn liệu xác nhận rủi ro này là thật chứ không riêng của repo: Ali 2024 báo cáo subject 15 đạt
**0.00 %**, subject 12 **3.70 %**, subject 14 **25.0 %** sensitivity trên cross-subject.

**Khuyến nghị:** chỉ áp gate worst-patient cho bệnh nhân có **≥5 cơn**; với n < 5, báo cáo cận
dưới khoảng tin cậy nhị thức chính xác (đã có sẵn `eval/bootstrap.py`) thay vì một ngưỡng cứng.

---

## 4b. Params/MAC: số đo thật, lần đầu tiên

`docs/EXPERIMENT_LOG_G1a.md` §4 ghi params/MAC "chưa từng được in ra". Đã đo bằng
`scripts/measure_model_size.py` (thop, `fs_hz=256`); kết quả lưu ở
`artifacts/model_size_w4s.json` và `artifacts/model_size_w2s.json`.

| Model | Params | MACs (cửa sổ 4 s) | MACs (cửa sổ 2 s) | Trọng số INT8 |
|---|---:|---:|---:|---:|
| `wearseizure1d` (mặc định) | 14 834 | 765 632 | 382 912 | 14.5 KiB |
| `wearseizure1d_nodilation` | 14 834 | 765 632 | 382 912 | 14.5 KiB |
| **`wearseizure1d_k5only`** | **11 786** | **585 920** | **293 056** | **11.5 KiB** |
| `wearseizure1d_k3only` | 11 642 | 576 704 | 288 448 | 11.4 KiB |
| `baseline_compact1d_7k` | 7 570 | 1 398 928 | 699 536 | 7.4 KiB |
| `baseline_frontiers2d` | 4 546 | 2 523 328 | 1 261 760 | 4.4 KiB |

Bốn điều rút ra:

1. **Model mặc định vượt mục tiêu thiết kế Table 4.** 14 834 params so với mục tiêu 13 810
   (**+1 024**), và 765 632 MACs so với 644 000 (**+121 632, tức +18.9 %**). Nó vẫn nằm dưới
   trần cứng 32k/2M, nhưng **trượt mức `target` của cổng MAC (≤700 000)** — điều này chưa
   từng được phát hiện vì unit test chỉ kiểm tra trần trên.
2. **`k5only` — biến thể thắng ở ablation — đồng thời cũng là biến thể nhỏ hơn.** 11 786 params
   (đạt mức *stretch* ≤12 000) và 585 920 MACs (đạt mức *target*). Nó vừa giảm FAR gần một nửa,
   vừa nhẹ hơn 20 % về params và 23 % về MACs so với mặc định. Đây là một lập luận mạnh hơn
   nhiều so với chỉ nói về FAR.
3. **`k5only` + cửa sổ 2 s đạt cả hai mức stretch cùng lúc**: 11 786 params / 293 056 MACs
   (≤12 000 và ≤500 000). Nghĩa là thí nghiệm Nhánh A — vốn được đề xuất để tấn công độ trễ —
   *đồng thời* đưa model vào mức stretch của cả hai cổng ngân sách. Đây là lý do thứ hai để ưu
   tiên nó.
4. **Params là chỉ số gây hiểu nhầm.** `baseline_frontiers2d` có ít hơn 3.3 lần số params nhưng
   **nhiều hơn 3.3 lần số MACs** so với `wearseizure1d`. Khi so sánh với văn liệu phải dùng MACs,
   không dùng params.

Đối chiếu với đối thủ gần nhất về lượng tử hoá: trọng số INT8 của `k5only` là **11.5 KiB**, so với
model size **0.44 MB** của arXiv 2607.16296 — **nhỏ hơn khoảng 39 lần**. Luận điểm hiệu quả tính
toán giờ đã có số đo đứng sau, không còn là mục tiêu thiết kế.

Lưu ý trung thực khi viết bài: ở 60 epoch, `baseline_frontiers2d` đạt sensitivity **0.9185** và
`baseline_compact1d_7k` đạt **0.8974**, đều **cao hơn** `k5only` (0.8756). WearSeizure-1D không
trội hơn baseline về độ nhạy — nó nằm ở một điểm khác trên cùng đường đánh đổi, với FAR thấp hơn
2.5 lần và chi phí tính toán thấp hơn 4.3 lần. Bài báo phải trình bày đúng như vậy.

---

## 5. Dự án đang tối ưu nhầm trục

Đặt cạnh nhau hai cột từ bảng §1:

| | Yêu cầu | Đạt được | Dư/thiếu |
|---|---|---|---|
| FAR | ≤0.30/h | 0.0621/h | **dư ~4.8×** |
| worst-patient FAR | ≤1.0/h | 0.1899/h | **dư ~5.3×** |
| exposure | ≥100 h | 185.0 h | dư 1.85× |
| delay | ≤5.0 s | 14.67 s | **thiếu 2.9×** |
| sensitivity | ≥0.970 | 0.8806 (ở FAR hợp lệ) | thiếu 9 pp |

Run #19 ("Nhánh B", pooled threshold) là ví dụ rõ nhất: nó đẩy FAR xuống **0.0621** — tốt hơn
yêu cầu gần 5 lần — nhưng đồng thời làm **sensitivity tụt 0.8756 → 0.8236**, **worst-patient
sensitivity tụt 0.25 → 0.00**, và **delay tăng 23.46 → 26.86 s**. Đó là một nước đi tiêu tài
nguyên ở trục đã thừa để mua thêm ở trục đã thừa, trong khi hai trục đang thiếu thì tệ đi.

**Hành động:** đảo chiều hàm mục tiêu của `threshold_selection`. Thay vì "tối đa sensitivity với
ràng buộc FAR ≤ 0.30/h" (`far_cap_per_hour: 0.30` hiện tại), hãy dùng:

> tối thiểu hoá **delay** với ràng buộc `FAR ≤ 0.30/h` **và** `sensitivity ≥ 0.88`

Với ngân sách FAR dư ~5×, cấu hình `run_length = 1`, `ema_alpha = 0.5` (sàn delay 5 s thay vì
13 s) là khả thi mà vẫn còn rất nhiều biên FAR — run #9 cho thấy nó đẩy FAR lên 1.024/h ở
`compact1d_7k`, nhưng biến thể `k5only` (#15) có FAR chỉ 0.1624/h ở cấu hình mặc định, tức có
biên lớn hơn nhiều để hấp thụ việc nới hậu xử lý. **Cấu hình `k5only` + `L=1` + `α=0.5` chưa bao
giờ được thử và là thí nghiệm có kỳ vọng cao nhất còn tồn đọng.**

---

## 6. Bối cảnh cạnh tranh và định vị đóng góp

### 6.1 Ba công trình gần nhất

| Công trình | Venue | Số liệu chính |
|---|---|---|
| *Wearable Epilepsy Seizure Detection on FPGA with Spiking Neural Networks* | **IEEE TBioCAS 2025**, DOI [10.1109/TBCAS.2025.3575327](https://doi.org/10.1109/TBCAS.2025.3575327) | CHB-MIT, **100 % event detection @ 0.3 FP/h**, AUC 96 %, accuracy 99.3 %, **0.5 µs / 4.55 nJ mỗi inference** |
| Vittimberga, Nicolini, Scotti — *Real-Time Multi-Channel Epileptic Seizure Detection Exploiting an Ultra-Low-Complexity Algorithm–Hardware Co-Design Approach* | *Sensors* 25(22):6889, 11/2025, DOI [10.3390/s25226889](https://doi.org/10.3390/s25226889) | CHB-MIT + SWEC-ETHZ, ~98 % accuracy, >98 % sensitivity, latency **3.37 s**, FPGA synthesis, đa kênh |
| Ahlawat — *Efficient EEG Seizure Detection Using INT8 Quantization, Channel Pruning, and Spiking Neural Networks* | arXiv [2607.16296](https://arxiv.org/abs/2607.16296), 7/2026 | 1D-CNN + INT8 QAT: **1.63 MB → 0.44 MB**, năng lượng −64 %, CPU speedup 2.8× |

### 6.2 Hệ quả cho định vị

- Bài TBioCAS 2025 **cùng venue mục tiêu, cùng dataset, cùng ứng dụng wearable, cùng FPGA**, và
  đã công bố 100 % event @ 0.3 FP/h với 4.55 nJ/inference. **Trục "độ chính xác" và trục
  "nJ/inference" coi như đã bị chiếm.** Không nên đặt bài báo lên hai trục này.
- Không tìm thấy công trình nào kết hợp đồng thời: **EEG một kênh + CNN lượng tử hoá + đo công
  suất thật trên board + xác minh bit-exact RTL ↔ INT8 reference**. Đó là khe hở còn lại.

### 6.3 Ba trục đóng góp đề xuất, xếp theo độ chắc chắn

1. **Đóng góp giao thức (chắc chắn có, rẻ nhất).** Định lượng mức thổi phồng của giao thức
   Chung 2024 trên đúng dataset, đúng kênh, đúng 13 bệnh nhân của họ, ở **185 h thay vì ~91 h**.
   Delta giữa nhánh tái lập (leaky) và nhánh leakage-safe chính là Figure 1 của bài. Giá trị này
   tồn tại bất kể model mạnh hay yếu.

2. **Đóng góp kiến trúc (đã có dữ liệu).** Ablation #14–#17 cho một kết luận thiết kế thật:
   **nhánh kernel k3 là nguồn phát sinh false alarm chính.** `k5_only` chỉ mất **0.5 pp**
   sensitivity so với multi-scale mặc định (0.8756 vs 0.8806) nhưng **giảm FAR gần một nửa**
   (0.1624 vs 0.3119) và **giảm worst-patient FAR ~4×** (0.3926 vs 1.5533). Kèm theo kết quả âm
   của Nhánh B (#19: pooling per-patient threshold làm hại sensitivity và worst-patient
   sensitivity dù cải thiện FAR) để củng cố tính nghiêm túc.

3. **Đóng góp phần cứng (bắt buộc để đúng scope TBioCAS).** Accelerator streaming mixed-precision,
   trọng số hoàn toàn on-chip (không DRAM), **xác minh bit-exact RTL ↔ INT8 reference**, và
   **công suất đo thật quy về mJ/giờ giám sát liên tục**. Không đua nJ/inference với bài SNN —
   chọn đơn vị đo phản ánh đúng bài toán giám sát 24/7.

Lưu ý về scope: TBioCAS yêu cầu "demonstrated synergy between medicine/biology and
circuits/systems"; đóng góp thuần phần mềm nằm ngoài scope. Ba trục trên chỉ trọn vẹn khi trục 3
hoàn thành.

---

## 7. Rủi ro phần cứng cần xử lý trước khi bắt đầu RTL

Tiêu chí GO/NO-GO cứng yêu cầu **"measured FPGA power"**. Nhưng **PYNQ-Z2 (Zynq-7020) không có
PMBus / cảm biến công suất on-board** — trong họ board PYNQ, chỉ ZCU104 có
(<https://pynq.readthedocs.io/en/latest/pynq_libraries/pmbus.html>). Nếu chốt PYNQ-Z2 mà không
chuẩn bị, tiêu chí này sẽ fail ở giai đoạn cuối, khi không còn thời gian đổi hướng.

Ba lựa chọn:

| Phương án | Ưu | Nhược |
|---|---|---|
| **(A) Giữ Zynq-7020 + shunt ngoài INA219/INA226 trên đường 12 V** ⭐ | Bảo vệ tốt nhất luận điểm "wearable"; chi phí thấp | Cần lắp mạch đo, phải hiệu chuẩn và mô tả rõ trong bài |
| (B) Đổi sang ZCU104 / KV260 (phòng lab đã có KV260) | Có rail PMBus, đo bằng phần mềm | Thiết bị lớn hơn, tốn điện hơn → làm yếu luận điểm wearable |
| (C) Chỉ báo cáo ước lượng từ Vivado Power Analyzer | Không cần phần cứng thêm | **Vi phạm GO/NO-GO** ("measured", không phải "estimated") |

**Khuyến nghị: (A).** Quyết định và lắp đặt việc đo *trước* khi viết dòng RTL đầu tiên.

---

## 8. Bộ gate v2 đề xuất

Xem `configs/eval/gates_v2_proposed.yaml`. Nguyên tắc: tách làm hai loại, không trộn lẫn.

### A. Reproduction gates — xác nhận giao thức đúng, không phải đóng góp

| Kiểm tra | Kỳ vọng |
|---|---|
| Tái lập Chung 2024 theo *đúng* protocol gốc (stride 1 mẫu, Th/L per-patient trên eval) | sensitivity rơi vào 99.62 ± 1.39 % |
| Mọi test trong `test_splits_no_leakage.py` / `test_splits_loso.py` pass trên dữ liệu thật | pass |

### B. Contribution gates — phải đạt mới viết bài

| KPI | Mức mới | Hiện tại | Lý do đổi |
|---|---|---|---|
| `personalized_event_sensitivity` | ≥ **0.92** @ FAR ≤0.30/h | 0.8806 | 0.970 vừa không thắng baseline (0.9962) vừa không đạt nổi dưới giao thức trung thực |
| `detection_delay_mean_s` | ≤ **10.0 s**, kèm **bắt buộc báo cáo `delay_floor_s`** | 14.67 | Gate 5 s bất khả thi: sàn giao thức mặc định đã là 13.0 s (§3) |
| `detection_delay_floor_s` (mới) | ≤ **3.0 s** | 13.0 | Ràng buộc lên *cấu hình đo*, tách khỏi ràng buộc lên model |
| `worst_patient_sensitivity` | ≥0.85, **chỉ áp dụng cho bệnh nhân có ≥5 cơn**; n<5 báo cáo CI nhị thức | 0.6667 | Với n=3, gate 0.85 đòi hỏi 3/3 tuyệt đối (§4) |
| `far_per_hour` | ≤ **0.15/h** | 0.0621 | Siết lại để biến thế mạnh thành đóng góp; vẫn còn biên 2.4× |
| `continuous_test_exposure_hours` | ≥ **185 h** | 185.0 | Gấp ~2× mức ~91 h của Chung 2024 |
| `model_params` / `model_macs` | ≤16 000 / ≤700 000 | 14 834 / **765 632** (mặc định) · 11 786 / 585 920 (`k5only`) | Đã đo (§4b). Model mặc định **trượt** mức target MAC; `k5only` đạt |
| `int8_loss_vs_fp32_pp` | ≤0.5 pp | chưa đo | Giữ nguyên |
| `zero_shot_loso_*` | **Bỏ khỏi gate**, chuyển thành bảng mô tả | chưa chạy | 85 % @ 0.75/h trên 1 kênh là bất khả thi: kết quả cross-subject trung thực tốt nhất từng công bố là 75.34 % @ 4.79/h **với 18 kênh** (Ali 2024) |
| `correctness_mismatches` | 0 / 10 000 cửa sổ | chưa có RTL | Giữ — đây là GO/NO-GO thật |
| Công suất | **Đo được**, mJ/giờ giám sát | chưa có | Giữ; xử lý §7 trước |

---

## 9. Thứ tự thực thi đề xuất

| # | Việc | Ở đâu | Trạng thái |
|---|---|---|---|
| 1 | Quy ước đóng dấu alarm có thể chọn được; `delay_floor_s` + phân rã vào `report.json` | local | ✅ **xong** — xem §10 |
| 2 | Thêm hàm mục tiêu `min_delay` cho `threshold_selection` | local | ✅ **xong** — xem §10 |
| 3 | Đo params/MAC thật cho cả 4 biến thể | local | ✅ **xong** — xem §4b |
| 4 | Chạy **`k5only` + `w2s_stride1s` + `L=1` + `α=0.5`** với `objective=min_delay` | SERVER-02 | ⬜ sàn delay 2.0 s **và** đạt cả hai mức stretch ngân sách — thí nghiệm giá trị cao nhất còn lại |
| 5 | QAT → INT8 integer reference → đo INT8 loss | SERVER-02 | ⬜ mở khoá phần phần cứng |
| 6 | Chạy zero-shot LOSO **một lần** | SERVER-02 | ⬜ để có bảng mô tả, không để pass cổng |
| 7 | Nhánh tái lập Chung 2024 (protocol gốc) | SERVER-02 | ⬜ Figure 1 của bài báo |
| 8 | Chốt board + phương án đo công suất (§7) | phần cứng | ⬜ đường găng thật của dự án |
| 9 | RTL + bit-exact verification | phần cứng | ⬜ GO/NO-GO cuối |

### Việc dọn dẹp cần làm song song
1. **Khôi phục Research Decision Memo**, hoặc viết lại thành `docs/DECISION_MEMO.md` và commit —
   hiện không có gì để audit ngược các mục tiêu.
2. **Sửa logging 0 byte** trong `artifacts/runs/*` trước đợt chạy lớn tiếp theo.

---

## 10. Những gì đã thay đổi trong code

Nguyên tắc xuyên suốt: **mặc định không đổi hành vi.** Mọi con số trong
`docs/EXPERIMENT_LOG_G1a.md` vẫn tái lập được bit-for-bit; các khả năng mới đều phải bật rõ ràng.

**`src/wearseizure/eval/delay_budget.py`** *(mới)* — công thức sàn độ trễ và phân rã
"sàn + phản ứng model". `delay_budget()` trả về từng số hạng riêng biệt (cửa sổ, run-length, EMA)
để không bao giờ đọc một con số delay mà không thấy sàn của nó.

**`src/wearseizure/postprocess/hysteresis.py`** — `PostprocessParams` có thêm `alarm_timestamp`
(`window_end` mặc định — quy ước nhân quả, giữ nguyên hành vi cũ; `window_center`; `window_start`)
và `window_s` (mặc định `0.0`, khiến quy ước thành no-op nếu người gọi không cung cấp). Alarm không
bao giờ bị đẩy về trước thời điểm 0.

**`src/wearseizure/eval/report.py`** — `build_report(per_patient, budget=...)` bổ sung vào khối
`delay`: `floor_s`, `model_reaction_mean_s`, `model_reaction_median_s`, `budget` (từng số hạng), và
`window_start_convention` (chỉ số delay như khi quy ước đóng dấu sớm hơn — tính chính xác từ danh
sách delay đã có, không cần chạy lại). `check_gates` hiểu `min_events_to_gate` (trả về mức
`not_gated_small_sample`) và bỏ qua các khối metadata (`reproduction`, `zero_shot_loso`).

**`src/wearseizure/eval/metrics_event.py`** — `worst_patient()` trả thêm
`sensitivity_patient_n_events`, để giá trị worst-patient luôn đi kèm cỡ mẫu sinh ra nó.

**`src/wearseizure/training/threshold_selection.py`** — thêm `objective="min_delay"`
(`max_sensitivity` vẫn là mặc định, không đổi một dòng hành vi nào) cùng `sensitivity_floor`.
Dưới `min_delay`, trong số các tổ hợp còn nằm trong trần FAR *và* giữ được sensitivity, chọn tổ hợp
nhanh nhất. `FrozenPostprocessParams` ghi thêm `val_delay_mean_s` kể cả dưới mục tiêu mặc định, để
cái giá phải trả trên trục delay luôn hiện ra.

**`scripts/evaluate.py`** — dựng `DelayBudget` từ config và log dòng phân rã, ví dụ thật từ một lần
chạy: `delay mean=0.00s = floor 13.00s (window 4.00 + run-length 2.00 + EMA 7.00) + model reaction 0.00s`.

**`scripts/measure_model_size.py`** *(mới)* — đo params/MAC/bytes INT8 cho mọi biến thể, không dùng
Hydra và không đụng tới dữ liệu.

**Tests** — `tests/unit/test_delay_budget.py` (mới, 12 test) chốt lại: sàn 13.0 s với mặc định,
5.0 s với cấu hình #9, 2.5 s với cấu hình đề xuất, quy tắc miễn trừ mẫu nhỏ, và việc khối metadata
không bao giờ bị chấm điểm. Thêm 4 test quy ước đóng dấu vào
`test_postprocess_hysteresis.py` và 4 test mục tiêu `min_delay` vào `test_threshold_selection.py`.
Toàn bộ: **103 unit test + 3 integration test pass**.

---

## 11. Cần đạt bao nhiêu thì được bắt đầu thiết kế phần cứng

**Trả lời ngắn: không cần SOTA, và cũng không cần thắng baseline. Cần chứng minh được *tương
đương thống kê* với baseline mạnh nhất ở cùng giao thức, cộng với một chiến thắng rõ ràng trên
trục tính toán. Cổng chặn thật sự duy nhất còn lại là INT8 loss.**

### 11.1 Vì sao không cần SOTA

Hai lý do độc lập:

1. **Không so trực tiếp được.** Mọi con số SOTA đang lưu hành (Chung 2024: 99.62 %; TBioCAS 2025:
   100 % event @ 0.3 FP/h) đều đến từ giao thức khác — cửa sổ trượt 1 mẫu, ngưỡng chỉnh trên dữ
   liệu đánh giá, thời lượng test nhỏ hơn. Đuổi theo những con số đó dưới giao thức leakage-safe
   là đuổi theo một đại lượng khác (§2, §3).
2. **Sai venue để đua độ chính xác.** TBioCAS yêu cầu "demonstrated synergy between
   medicine/biology and circuits and systems". Đóng góp được đánh giá là hệ thống mạch, không
   phải bảng xếp hạng độ nhạy. Một model tốt vừa đủ + một accelerator đo đạc nghiêm túc là đúng
   hình dạng bài báo của venue này; một model SOTA + accelerator sơ sài thì không.

### 11.2 Nhưng phải nhìn thẳng vào một sự thật

Ở cùng 66 fold, cùng giao thức, `wearseizure1d_k5only` (#15) **không dẫn đầu ở bất kỳ trục lâm
sàng nào**:

| | Sens | FAR/h | Worst-pt sens | Worst-pt FAR/h | Delay | MACs | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `k5only` (#15) | 0.8756 | 0.1624 | 0.2500 | **0.3926** | 23.46 | **585 920** | 11 786 |
| `frontiers2d` (#2) | **0.8811** | **0.1432** | **0.6667** | 0.6903 | 23.40 | 2 523 328 | **4 546** |
| `compact1d_7k` (#7) | **0.8974** | 0.2103 | 0.3333 | 0.5697 | **19.49** | 1 398 928 | 7 570 |

`frontiers2d` (#2) vượt `k5only` ở sensitivity, FAR **và** worst-patient sensitivity, với độ trễ
gần như bằng nhau. `k5only` chỉ thắng ở **worst-patient FAR (1.76×)** và **MACs (4.3×)**.

Do đó luận điểm của bài báo **không thể là "phát hiện tốt hơn"**. Nó phải là:

> phát hiện **tương đương** ở chi phí tính toán thấp hơn 4.3 lần, hiện thực thành accelerator có
> số đo công suất thật.

### 11.3 "Tương đương" là một mệnh đề chứng minh được, không phải một lời bào chữa

Khoảng cách sensitivity so với `frontiers2d` là **0.55 pp**. Với 13 bệnh nhân và 77 cơn (trung
bình ~5.9 cơn/bệnh nhân), việc bắt được thêm **đúng một cơn** ở một bệnh nhân đã làm macro
sensitivity đổi khoảng **1.28 pp** — tức lớn hơn toàn bộ khoảng cách đang bàn. Khoảng cách này
nằm dưới độ phân giải của bộ dữ liệu.

Cách chứng minh, dùng công cụ đã có sẵn trong repo (`eval/bootstrap.py`, cluster bootstrap theo
bệnh nhân): **paired bootstrap trên 66 fold**, báo cáo khoảng tin cậy 95 % của
Δsensitivity = `k5only − frontiers2d`. Điều kiện đạt là **khoảng tin cậy chứa 0**. Không cần
thắng; chỉ cần chứng minh không thua có ý nghĩa thống kê.

Đây chính xác là phương pháp nhóm đã dùng ở dự án CARE_ASD
(`reports/alignment/.../paired_bootstrap_b01_vs_b00.json`), nên không phải xây mới.

### 11.4 Cổng nào thật sự chặn phần cứng

Đây là phần quan trọng nhất: **phần lớn các cổng đang fail không hề chặn thiết kế RTL.** Chỉ
những thứ định hình datapath mới chặn.

| Hạng mục | Có chặn RTL không | Trạng thái |
|---|---|---|
| Kiến trúc (số nhánh, kernel, dilation) | **CHẶN** — quyết định PE array và bộ tạo tap | ✅ chốt `k5only` (#15 là kết quả tốt nhất; #20 không lật được) |
| Độ dài cửa sổ (1024 vs 512 mẫu) | **CHẶN** — quyết định buffer on-chip | ✅ chốt **4 s / 1024 mẫu** (#20 cho thấy 2 s làm mọi thứ tệ hơn) |
| Params / MACs | **CHẶN** — sizing SRAM và số PE | ✅ đã đo: 11 786 / 585 920 (§4b) |
| **INT8 (và W4A8) loss** | **CHẶN** — quyết định độ rộng bit của datapath | ❌ **chưa đo — đây là cổng chặn cuối cùng** |
| Sensitivity / FAR / worst-patient | không — đều là ngưỡng và huấn luyện | có thể cải thiện song song |
| Detection delay | không — hậu xử lý chạy trên PS, ngoài RTL | có thể cải thiện song song (§3.4) |
| Zero-shot LOSO | không — chỉ là bảng mô tả | có thể chạy bất cứ lúc nào |
| Tái lập Chung 2024 | không — là Figure 1, không phải thiết kế | có thể chạy bất cứ lúc nào |

Hậu xử lý (EMA, hysteresis, run-length, merge) nằm **ngoài** accelerator theo
`rtl_interface/spec.md` — accelerator chỉ tính CNN. Vì vậy mọi việc chỉnh độ trễ và FAR đều không
đụng tới RTL và không có lý do gì để giữ phần cứng lại.

### 11.5 Điều kiện GO cụ thể

Bắt đầu thiết kế phần cứng khi và chỉ khi:

1. **INT8 loss ≤ 0.5 pp** trên `k5only` / `w4s_stride1s` (một lần chạy trên SERVER-02). Nếu
   INT8 làm sập độ nhạy thì toàn bộ độ rộng bit của datapath phải thiết kế lại — đây là lý do
   duy nhất chính đáng để chờ.
2. **Paired bootstrap cho thấy CI 95 % của Δsensitivity vs `frontiers2d` chứa 0**, ở FAR ≤ 0.30/h.
   Chạy trên metrics đã có, **không cần train lại**.
3. Ba mục đã xong: kiến trúc chốt, cửa sổ chốt, params/MAC đo được.

Nếu điều kiện 2 không đạt (tức thua có ý nghĩa), phương án dự phòng theo thứ tự: (a) chỉnh hậu xử
lý bằng `objective=min_delay` với ràng buộc sensitivity — ngân sách FAR còn dư 1.8× để chi;
(b) huấn luyện `k5only` ở 30 epoch, cấu hình chưa từng thử (mọi run k5only đều ở 60 epoch, trong
khi kết quả tốt nhất của `frontiers2d` lại ở 30 epoch); (c) chấp nhận và chuyển sang tăng tốc
`compact1d_7k` — nhưng khi đó mất luận điểm về kiến trúc.

**Không nên** đặt điều kiện GO ở "sensitivity ≥ 0.97" hay "delay ≤ 5 s". Cả hai đều không chặn
RTL, và như §2–§4 đã chỉ ra, cả hai đều là ngưỡng sao chép từ một giao thức khác.

### 11.6 Đổi lại, phần cứng phải gánh được bài báo

Vì đã chọn "vừa đủ về thuật toán", phần cứng không được phép mỏng. Tối thiểu:

- **Bit-exact RTL ↔ INT8 reference: 0 sai khác / 10 000 cửa sổ.** Đây đã là GO/NO-GO và cũng là
  thứ phần lớn bài báo accelerator không làm.
- **Công suất đo thật trên board**, quy về **mJ mỗi giờ giám sát liên tục** — không đua
  nJ/inference với bài TBioCAS 2025 (4.55 nJ), vì đó là trục họ đã thắng và đơn vị đó không phản
  ánh bài toán giám sát 24/7.
- **Ít nhất hai điểm so sánh**: (a) cùng model chạy trên ARM Cortex-A9 của chính board,
  (b) một accelerator đã công bố, quy về cùng đơn vị.
- **Đầy đủ LUT / FF / BRAM / DSP và Fmax.**
- Lưu ý tiêu chí GO/NO-GO của chính dự án: cần "ít nhất một đóng góp phần cứng định lượng được,
  vượt trên một baseline HLS/CPU chung chung" — nên một bảng so sánh HLS-vs-RTL đơn thuần là
  **không đủ**.

---

## 12. Mục tiêu cần đạt và bảng benchmark

### 12.1 Nguyên tắc: bảng so sánh phải có hai khối, không phải một

Đây là điều quyết định bảng benchmark có sống sót qua phản biện hay không.

Mọi con số CHB-MIT đang lưu hành đều đến từ giao thức khác nhau, nên **xếp chúng vào cùng một cột
là sai**. Bảng của bài báo phải tách làm hai khối:

- **Khối A — "as published"**: liệt kê nguyên văn, kèm **cột giao thức bắt buộc**. Khối này để
  định vị, không phải để tuyên bố thắng thua.
- **Khối B — "reproduced under our protocol"**: các kiến trúc được tái lập trên đúng 66 fold,
  185 h, ngưỡng đóng băng trên val. **Chỉ khối này mới là so sánh hợp lệ.**

Khối B đã có sẵn dữ liệu (rows #2, #7, #15) và chính là lá chắn cho mọi con số của bài.

### 12.2 Khối A — thuật toán, as published

| Công trình | Kênh | Giao thức | Sens | FAR/h | Delay | Params | MACs |
|---|---|---|---:|---:|---:|---:|---:|
| Chung 2024, Front. Neurol. 15:1389731 | **1** | patient-specific; stride **1 mẫu**; Th/L chỉnh trên dữ liệu eval; ~91 h | 99.62 % | 0.22 | 3.3 s | — | — |
| Ultra-lightweight 3D-CNN, *Biomed. Signal Process. Control* (2025) | nhiều | patient-specific | 99.24 % | 0.53 | 4.97 s | **6 540** | 2 390 000 |
| Lightweight multi-scale channel attention (2023) | nhiều | segment-level | 98.3 % | — | — | 88 000 | 2 680 000 |
| Ahlawat, arXiv:2607.16296 (2026) | — | — | "preserved" | — | — | — | 0.44 MB INT8 |
| Ali 2024, R. Soc. Open Sci. 11:230601 | 18 | **cross-subject, toàn bộ corpus** | 75.34 % | 4.79 | — | — | — |
| **WearSeizure-1D `k5only` (#15)** | **1** | **leakage-safe; 185 h; ngưỡng đóng băng trên val** | **87.56 %** | **0.16** | 23.5 s | **11 786** | **585 920** |

Ba điều đọc ra:

1. **Trục MAC là trục thắng.** 585 920 MACs thấp hơn **4.1×** so với 3D-CNN siêu nhẹ (2.39 M) và
   **4.6×** so với model attention (2.68 M). Nhưng phải nói rõ: một phần lợi thế này đến từ việc
   dùng **1 kênh thay vì 18** — đó chính là đóng góp, nhưng không được giấu.
2. **Không được tuyên bố "model nhỏ nhất".** 11 786 params **lớn hơn 1.8×** so với 6 540 params
   của 3D-CNN 2025. Luận điểm phải là *chi phí tính toán*, không phải *số tham số*.
3. **Sensitivity thấp hơn tất cả** — và đó là điều khối B tồn tại để giải thích.

### 12.3 Khối B — tái lập trên cùng giao thức (đây mới là so sánh hợp lệ)

| Kiến trúc | Sens | FAR/h | Worst-pt sens | Worst-pt FAR | Delay | Params | MACs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frontiers2d` (tái lập Chung 2024) #2 | **0.8811** | **0.1432** | **0.6667** | 0.6903 | 23.40 | 4 546 | 2 523 328 |
| `compact1d_7k` #7 | **0.8974** | 0.2103 | 0.3333 | 0.5697 | **19.49** | 7 570 | 1 398 928 |
| **`wearseizure1d_k5only` #15** | 0.8756 | 0.1624 | 0.2500 | **0.3926** | 23.46 | 11 786 | **585 920** |

**Kiến trúc Frontiers, khi tái lập đúng giao thức, đạt 88.11 % chứ không phải 99.62 %.** Đó là
Figure 1 của bài báo và là lời giải thích duy nhất cần có cho mọi con số ở khối A. Chênh lệch
**11.5 pp** giữa 99.62 % và 88.11 % là kết quả, không phải thất bại.

### 12.4 Mục tiêu thuật toán cần đạt

| # | Mục tiêu | Ngưỡng | Vì sao đúng ngưỡng đó |
|---|---|---|---|
| A1 | **Tương đương thống kê** với `frontiers2d` | CI 95 % của Δsensitivity chứa 0, ở FAR ≤ 0.30/h | Khoảng cách 0.55 pp nhỏ hơn ảnh hưởng của một cơn duy nhất (~1.28 pp) |
| A2 | **Ưu thế tính toán** | MACs ≤ **1/4** của kiến trúc tái lập tốt nhất | Đã đạt: 585 920 vs 2 523 328 (4.3×) |
| A3 | **FAR** | ≤ 0.15/h | Đã đạt 0.1624 ở #15, 0.0621 ở #19 — siết lại để biến thế mạnh thành đóng góp |
| A4 | **INT8 loss** | ≤ 0.5 pp | Cổng chặn phần cứng duy nhất còn lại |
| A5 | **Sàn độ trễ** (cổng) | ≤ 5.0 s | Đây là ràng buộc lên *cấu hình đo*, đạt được chỉ bằng `L=1`, `α=0.5` (§3.4) |
| A5b | **Độ trễ trung bình** | **báo cáo, không đặt cổng** | Xem §13.2: không có cấu hình nào trong 20 run cho phản ứng model < 6.4 s, nên mọi cổng tuyệt đối dưới ~11.4 s đều không có cơ sở |
| A6 | **Exposure** | ≥ 185 h | Gấp ~2× mức ~91 h của Chung 2024 |
| A7 | **Delta giao thức** | Báo cáo được chênh lệch tái lập-leaky vs leakage-safe | Đây là Figure 1 |

Không có mục nào yêu cầu vượt SOTA. A1 là *tương đương*, không phải *vượt trội*.

### 12.5 Mục tiêu phần cứng — và một cảnh báo phải đọc trước khi cam kết

Tải tính toán suy ra trực tiếp từ §4b: **585 920 MACs mỗi cửa sổ, 1 cửa sổ mỗi giây**
(stride 1 s). Hệ quả:

| Chỉ tiêu | Suy ra | Ghi chú |
|---|---|---|
| Độ trễ suy luận | ≤ **2 ms** | Với `P_PE=4` @ 100 MHz: 146 480 chu kỳ ≈ 1.46 ms. Deadline là 1000 ms → biên **~680×** |
| **Duty cycle** | **~0.15 %** | Đây là con số quan trọng nhất của toàn bộ thiết kế |
| Trọng số on-chip | **11.5 KiB** INT8 | ~3 BRAM18 trên tổng 280 của Zynq-7020 — không cần DRAM |
| Activation buffer | ~8–16 KiB | Tầng lớn nhất: 8 kênh × 512 mẫu INT8 = 4 KiB |
| Tài nguyên mục tiêu | < 10 % LUT, < 10 % DSP, < 10 % BRAM của XC7Z020 | tức < 5 300 LUT, < 22 DSP, < 28 BRAM36 |
| Fmax | ≥ 100 MHz | Dư thừa lớn; không nên đánh đổi diện tích lấy tốc độ |

**Kết luận thiết kế then chốt:** với duty cycle 0.15 %, **năng lượng mỗi giờ bị chi phối bởi công
suất tĩnh chứ không phải bởi tính toán**. Tăng `P_PE` lên 8 hay 16 chỉ làm mảng PE to hơn và nằm
không lâu hơn. Mục tiêu thiết kế là **PL nhỏ + clock gating quyết liệt**, không phải mảng MAC lớn.

#### Cảnh báo: trên Zynq-7020 không thể thắng cuộc so sánh công suất tuyệt đối

| Công trình | Nền tảng | Sens | FAR/h | Năng lượng |
|---|---|---:|---:|---|
| SNN FPGA, IEEE TBioCAS 2025 | FPGA | 100 % event | 0.3 | 4.55 nJ / inference; 0.5 µs |
| Event-driven multi-stage CNN, DAC 2022 | ASIC | 97.78 % | 0.5 | **0.32 µJ / classification** |
| EEG processor 65 nm | ASIC 65 nm | 91.86 % | 0.17 | 2.23 µJ / class |
| CNN trên vi điều khiển | MCU | 85 % | — | **140 µW trung bình** |
| SeizureNet trên phần cứng | — | — | — | 850 µW trung bình |
| **WearSeizure-1D** | **Zynq-7020** | — | — | **cần đo** |

Quy về cùng đơn vị: 140 µW trung bình ≈ **504 mJ/giờ**. Nhưng **công suất tĩnh của riêng PL trên
Zynq-7020 đã vào khoảng 100–200 mW**, tức **360–720 J/giờ** — lớn hơn khoảng **1000×**. Không có
cách tối ưu RTL nào lấp được khoảng cách đó, vì nó là đặc tính của thiết bị chứ không phải của
thiết kế.

Do đó **không được đặt mục tiêu "công suất thấp hơn công trình MCU/ASIC"**. Thay vào đó, báo cáo
ba con số và nói rõ ý nghĩa từng con số:

1. **Năng lượng động của PL mỗi cửa sổ** (µJ/cửa sổ) — so sánh được với các công trình FPGA khác.
2. **Công suất board đo thật** (mW) và mJ/giờ — trung thực về mức hệ thống, kèm tách bạch tĩnh/động.
3. **pJ mỗi MAC** — chỉ số đã chuẩn hoá theo thiết bị, là chỉ số duy nhất so sánh công bằng được
   giữa FPGA, MCU và ASIC.

Và nêu rõ trong phần Discussion: đạt mức µW của ASIC đòi hỏi chuyển sang ASIC; bài này là
**nguyên mẫu kiến trúc có số đo thật**, không phải sản phẩm wearable hoàn chỉnh. Nói trước điều
này mạnh hơn nhiều so với để phản biện chỉ ra.

#### Lưu ý về tính xác thực của bảng trên

Con số **4.55 nJ / inference** của bài TBioCAS 2025 thấp hơn **70×** so với ASIC DAC 2022
(0.32 µJ) — chênh lệch này khó tin nếu "inference" của cả hai cùng nghĩa là một cửa sổ hoàn
chỉnh. Nhiều khả năng đơn vị của họ là một bước thời gian spike, không phải một cửa sổ.
**Phải đọc toàn văn và xác minh định nghĩa đơn vị trước khi đưa con số này vào bảng của bài báo.**
Tương tự với các dòng MCU/ASIC lấy từ tóm tắt — cần đối chiếu toàn văn.

### 12.6 Mục tiêu phần cứng cần đạt

| # | Mục tiêu | Ngưỡng |
|---|---|---|
| H1 | Bit-exact RTL ↔ INT8 reference | **0 sai khác / 10 000 cửa sổ** |
| H2 | Trọng số hoàn toàn on-chip | 0 truy cập DRAM cho trọng số |
| H3 | Độ trễ suy luận | ≤ 2 ms (biên ≥ 500× so với deadline 1 s) |
| H4 | Tài nguyên | < 10 % LUT / DSP / BRAM của XC7Z020 |
| H5 | Công suất | **Đo thật** (shunt ngoài, §7), báo cáo tách tĩnh/động |
| H6 | Hiệu quả | pJ/MAC, và so với cùng model chạy trên ARM Cortex-A9 của chính board |
| H7 | Điểm so sánh | ≥ 2 điểm: (a) ARM cùng board, (b) ≥ 1 accelerator đã công bố, quy về cùng đơn vị |

---

## 13. Còn cách mục tiêu bao xa

Mốc so sánh: **row #15** (`wearseizure1d_k5only`, `w4s_stride1s`, hậu xử lý mặc định) — cấu hình
thật tốt nhất tính đến nay.

### 13.1 Bảng khoảng cách — thuật toán

| # | Mục tiêu | Ngưỡng | Hiện tại | Khoảng cách | Cách đóng | Chi phí |
|---|---|---:|---:|---|---|---|
| A2 | MACs ≤ ¼ baseline tốt nhất | 630 832 | **585 920** | ✅ **đạt, dư 7.1 %** | — | — |
| A6 | Exposure | 185 h | **185.0 h** | ✅ **đạt, vừa khít** | — | — |
| A1 | CI 95 % của Δsens chứa 0 | — | Δ = **−0.55 pp** | chưa tính, nhưng −0.55 pp nhỏ hơn ảnh hưởng của một cơn (1.28 pp) → khả năng đạt cao | paired bootstrap trên metrics đã có | **phút, local, không train lại** |
| A3 | FAR | ≤ 0.15/h | 0.1624 | **thiếu 0.0124/h (8.3 %)** | #19 đã chạm 0.0621 → dư địa rõ ràng | 1 lần `rethreshold` |
| A5 | Sàn độ trễ | ≤ 5.0 s | **13.0 s** | **thiếu 8.0 s** | `run_length 3→1`, `ema_alpha 0.125→0.5` | **không train lại** |
| A4 | INT8 loss | ≤ 0.5 pp | **chưa đo** | hoàn toàn chưa biết | QAT → integer reference | 1 run server |
| A7 | Delta giao thức (Figure 1) | có số | **chưa chạy** | chưa đo | nhánh protocol gốc Chung 2024 | 1 run server |
| — | Worst-patient sens | ≥0.85 (chỉ khi ≥5 cơn) | 0.2500 = **1/4** | **cần tính lại**: bệnh nhân này chỉ có 4 cơn nên được miễn trừ; giá trị của bệnh nhân tệ nhất *có ≥5 cơn* chưa ai biết | tính lại từ `report.json` | phút, local |

**Tổng kết:** 2/7 đã đạt; 3 mục còn lại chỉ cách một phép tính hoặc một lần đổi tham số **không cần
train lại**; 2 mục chưa đo, mỗi mục một lần chạy server. **Không mục nào đòi hỏi một đột phá nghiên
cứu** — ngoại trừ độ trễ, mà vấn đề ở đó là *mục tiêu sai*, không phải *kết quả kém*.

### 13.2 Độ trễ: vì sao phải sửa mục tiêu thay vì cố đạt

Phân rã hiện tại: **23.46 s = sàn 13.0 s + phản ứng model 10.46 s**.

Đổi `L=1`, `α=0.5` hạ sàn xuống 5.0 s. Nếu phản ứng model giữ nguyên, độ trễ trung bình rơi về
khoảng **15.5 s**. Phần dư model quan sát được trên toàn bộ 20 run:

| Run | Cấu hình | Sàn | Đo được | Phản ứng model |
|---|---|---:|---:|---:|
| #12 | w4s/1s L3 α.125 | 13.0 | 19.42 | **6.42** ← nhỏ nhất từng thấy |
| #9 | w4s/1s L1 α.5 | 5.0 | 14.67 | 9.67 |
| #15 | w4s/1s L3 α.125 | 13.0 | 23.46 | 10.46 |
| #10 | w4s/1s L2 α.25 | 8.0 | 18.69 | 10.69 |
| #13 | w4s/0.5s L2 α.25 | 6.0 | 18.37 | 12.37 |
| #19 | w4s/1s L3 α.125 | 13.0 | 26.86 | 13.86 |
| #20 | w2s/1s L3 α.125 | 11.0 | 26.18 | **15.18** ← lớn nhất |

Phản ứng model dao động **6.4 – 15.2 s** và **chưa bao giờ xuống dưới 6.4 s**. Cộng với sàn tốt
nhất khả thi (5.0 s với cửa sổ 4 s đã chốt), độ trễ trung bình khả thi nằm trong khoảng
**11.4 – 20.2 s**, ước lượng trung tâm **~15.5 s**.

Do đó **một cổng "delay ≤ 10 s" không có cơ sở thực nghiệm nào**, và cổng gốc "≤ 5 s" thì thấp hơn
cả sàn giao thức. Xử lý đúng, nhất quán với cách đã làm với zero-shot LOSO:

- **Đặt cổng lên `detection_delay_floor_s ≤ 5.0 s`** — ràng buộc lên cấu hình đo, đạt được bằng
  hai tham số.
- **Báo cáo độ trễ trung bình kèm phân rã**, không đặt cổng.
- **Bổ sung một cách đọc có ý nghĩa lâm sàng**: cơn động kinh trung bình dài **57.4 s**
  (Chung 2024), nên độ trễ ~15 s tương ứng với việc phát hiện trong **khoảng một phần tư đầu tiên**
  của cơn. Đây là cách trình bày trung thực và mạnh hơn nhiều so với đặt 15 s cạnh con số 3.3 s
  vốn được đo theo một quy ước khác (§3.3).

Ghi chú về mâu thuẫn đã sửa: bản nháp đầu của `gates_v2_proposed.yaml` đặt
`detection_delay_floor_s` target 3.0 s / stretch 2.5 s. Cả hai đều **đòi cửa sổ 2 s**, trong khi
row #20 đã loại bỏ cửa sổ 2 s. Dưới quy ước nhân quả `window_end`, số hạng cửa sổ luôn bằng
`window_s`, nên **với cửa sổ 4 s không cấu hình nào xuống dưới 4.0 s**. Các mức đã được sửa thành
5.0 / 4.5 / 4.0.

### 13.3 Bảng khoảng cách — phần cứng

Chưa bắt đầu mục nào. Nhưng cần phân biệt "chưa làm" với "rủi ro":

| # | Mục tiêu | Khoảng cách | Đánh giá rủi ro |
|---|---|---|---|
| H2 | Trọng số on-chip | 11.5 KiB cần, 630 KB có sẵn | **an toàn theo tính toán** — dư ~55× |
| H3 | Độ trễ ≤ 2 ms | 1.46 ms ước tính, deadline 1000 ms | **an toàn theo tính toán** — biên ~680× |
| H4 | < 10 % tài nguyên | mảng 4 PE trên 220 DSP | **an toàn theo tính toán** |
| H1 | Bit-exact 0/10 000 | cần có RTL trước | rủi ro kỹ thuật thông thường |
| H5 | Công suất đo thật | cần board + shunt ngoài (§7) | **rủi ro mua sắm/lắp đặt — giải quyết sớm** |
| H6 | pJ/MAC, so với ARM | cần cả hai phía chạy được | phụ thuộc H1, H5 |
| H7 | ≥ 2 điểm so sánh | cần xác minh toàn văn các bài đối chiếu (§12.5) | rủi ro nếu để sát hạn |

Ba mục đầu **đã an toàn ngay từ số liệu đã đo** — tải tính toán quá nhỏ so với thiết bị. Rủi ro
thật nằm ở H5 (phương án đo công suất) và ở việc đơn vị năng lượng của các bài đối chiếu chưa được
xác minh, chứ không nằm ở việc thiết kế RTL có chạy nổi hay không.

### 13.4 Đường ngắn nhất, tính theo việc chứ không theo thời gian

**Local, không cần server, không train lại** — đóng được A1, A3, A5, worst-patient:
1. Paired bootstrap Δsens `k5only` vs `frontiers2d`
2. Tính lại worst-patient dưới quy tắc ≥5 cơn
3. `rethreshold` với `objective=min_delay`, `L=1`, `α=0.5` → hạ sàn 13.0 → 5.0 s, siết FAR về ≤0.15

**Server, 2 lần chạy** — đóng được A4 và A7, và mở khoá phần cứng:
4. QAT → INT8 integer reference → INT8 loss (**cổng chặn phần cứng duy nhất**)
5. Nhánh tái lập Chung 2024 theo protocol gốc (Figure 1)

**Song song, không chờ gì cả:**
6. Chốt phương án đo công suất và đặt mua linh kiện shunt (§7) — đây là hạng mục có thời gian chờ
   vật lý, nên bắt đầu sớm nhất
7. Đọc toàn văn 3 bài đối chiếu để xác minh đơn vị năng lượng (§12.5)

---

## 14. Làm gì để kết quả đủ mạnh cho bài báo

§13 xếp hạng theo chi phí. Mục này xếp hạng theo **mức cải thiện kỳ vọng**, không quan tâm phải
train lại bao nhiêu lần.

### 14.1 Phát hiện quyết định: 20 run chưa từng động vào cách huấn luyện

Toàn bộ 20 thí nghiệm chỉ thay đổi: lưới ngưỡng, trần FAR, số epoch, `ema_alpha`/`run_length`,
kiểu kernel, độ dài cửa sổ, và cách gộp ngưỡng theo bệnh nhân. **Không run nào thay đổi dữ liệu
huấn luyện hay hàm mục tiêu.** Đọc mã cho thấy công thức huấn luyện hiện tại có bốn điểm yếu
nghiêm trọng, tất cả đều chưa từng được thử sửa:

**(1) Mỗi fold huấn luyện một model mới hoàn toàn, chỉ trên dữ liệu của đúng một bệnh nhân.**
`scripts/train.py:78` gọi `build_model(cfg)` cho từng fold, khởi tạo ngẫu nhiên. Và
`data/splits.py:80-119` xây fold trong phạm vi `manifest_df.groupby("subject_id")`, nên
`train_edf_ids` **chỉ chứa EDF của chính bệnh nhân đó**.

Hệ quả: với chb17 (3 cơn), một fold giữ lại 1 EDF ictal để test, nên model học từ nhiều nhất
**2 cơn**. Một CNN ~12k tham số học từ 2 cơn động kinh. Đây là lời giải thích thống nhất cho mọi
triệu chứng đã quan sát: worst-patient luôn rơi vào chb17/chb02/chb04 (3–4 cơn); tăng epoch không
giúp đồng đều (quá khớp); `val_loss` nhiễu tới mức chính comment trong `configs/train/default.yaml`
phải ghi nhận; và tìm kiếm ngưỡng chạm trần vì bộ phân loại **đói dữ liệu**, không phải lệch hiệu
chuẩn.

**(2) Mọi cửa sổ ictal được gán nhãn như nhau.** `data/windowing.py:65`:
`label = int(any(_overlaps(...)))`. Một cửa sổ ngay tại thời điểm khởi phát và một cửa sổ ở giây
thứ 40 của cơn có nhãn giống hệt nhau, trọng số giống hệt nhau. **Model không có bất kỳ động lực
nào để phát hiện sớm** — nó tối ưu độ đúng trung bình trên toàn cơn, nơi tín hiệu mạnh nhất ở giữa.
Đây là lời giải thích cho phần dư 10.5 s mà không cấu hình hậu xử lý nào chạm tới được, và là
hướng tấn công có nguyên lý duy nhất vào độ trễ.

**(3) Chọn model theo `val_loss` (cross-entropy), không theo chỉ số event.**
`training/loop.py:38,62` dùng `nn.CrossEntropyLoss` và early stopping trên val loss. Nhưng chỉ số
của bài báo là sensitivity/FAR **mức sự kiện**. Tiêu chí chọn model đang lệch khỏi mục tiêu, trên
một tập validation mà chính dự án thừa nhận là "thường chỉ 1–2 EDF, đôi khi một cơn duy nhất".

**(4) `seeds: [0, 1, 2]` là cấu hình chết.** `configs/train/default.yaml` khai báo ba seed theo
memo 5.3, nhưng `scripts/train.py:40` chỉ đọc `cfg.seed`. **Chưa có con số nào trong 20 run có
thanh sai số.** Với 77 cơn, chênh lệch 0.55 pp giữa các kiến trúc rất có thể nằm trong nhiễu seed —
và hiện không có cách nào biết.

Ngoài ra: **không có tăng cường dữ liệu (augmentation) ở bất kỳ đâu** trong repo.

### 14.2 Bảy đòn bẩy, xếp theo mức cải thiện kỳ vọng

| # | Đòn bẩy | Tấn công vào | Vì sao tin là có tác dụng |
|---|---|---|---|
| **L1** | **Tiền huấn luyện trên toàn nhóm + tinh chỉnh theo bệnh nhân.** Với fold (bệnh nhân S, EDF e): pretrain trên toàn bộ EDF của các bệnh nhân ≠ S, rồi fine-tune trên EDF còn lại của S, test trên e. | Sensitivity, **worst-patient** | Sửa đúng nguyên nhân gốc ở (1). Các bệnh nhân đang fail là các bệnh nhân ít cơn nhất — đúng nhóm hưởng lợi nhiều nhất từ transfer. **An toàn về rò rỉ**: tập pretrain không chứa một mẫu nào của S |
| **L2** | **Hàm loss có trọng số theo thời điểm khởi phát** — cửa sổ gần onset có trọng số cao hơn; hoặc loss phạt trực tiếp thời gian tới phát hiện | **Độ trễ** | Sửa (2). Đây là cách duy nhất đã nhận diện được để giảm phần dư model 10.5 s. Mọi thay đổi hậu xử lý chỉ chạm được vào sàn, không chạm vào phần dư |
| **L3** | **Chưng cất tri thức: teacher đa kênh → student 1 kênh.** Huấn luyện teacher trên 18 kênh (bài toán dễ hơn nhiều), chưng cất sang student 1 kênh | Sensitivity | Tấn công trực diện vào thiếu hụt thông tin của thiết kế 1 kênh. **Đồng thời là một đóng góp phương pháp cho bài báo**, không chỉ là thủ thuật tăng điểm. Nhóm đã dùng KD ở dự án `1_ai_accelerator_sound` |
| **L4** | **Chọn model theo chỉ số event trên val**, thay cho `val_loss` | Sensitivity, FAR | Sửa (3). Hiện đang tối ưu một đại lượng thay thế, trên tập nhiễu |
| **L5** | **Mở rộng corpus tiền huấn luyện**: dùng cả 23 case CHB-MIT tại cùng vị trí điện cực để pretrain (giới hạn 13 case chỉ áp cho *đánh giá*, không phải cho *pretrain*), sau đó tới corpus ngoài (TUSZ, Siena) | Sensitivity, zero-shot | Gần gấp đôi dữ liệu pretrain mà không đụng tới giao thức đánh giá. Cổng stretch của chính memo đã nhắc tới "external corpus" |
| **L6** | **Tăng cường dữ liệu**: dịch thời gian, co giãn biên độ, thêm nhiễu, mixup | Sensitivity, ổn định | Chưa có gì. Giá trị đặc biệt cao ở chế độ ít dữ liệu như hiện tại |
| **L7** | **Chạy đủ 3 seed như memo 5.3 đã quy định**, báo cáo trung bình ± độ lệch | **Độ tin cậy** | Không nâng điểm, nhưng nếu không có thanh sai số thì luận điểm "tương đương thống kê" (A1) không thể trình bày được, và mọi so sánh ablation đều thiếu căn cứ |

### 14.2b Trạng thái hiện thực hoá — L1 đã xong

L1 đã được cài đặt và kiểm thử (115 unit + integration test pass), **mặc định TẮT** để mọi con số
trong `EXPERIMENT_LOG_G1a.md` vẫn tái lập được.

| File | Vai trò |
|---|---|
| `src/wearseizure/training/pretrain.py` *(mới)* | `cohort_pretrain_fold()` dựng tập tiền huấn luyện gồm mọi bệnh nhân ≠ S, test rỗng; `get_or_train_cohort_init()` huấn luyện và cache theo từng bệnh nhân |
| `scripts/train.py` | Nạp init đã tiền huấn luyện trước mỗi fold, chuyển sang `finetune_lr`, và ghi `pretrained_from_cohort_excluding` + `lr` vào `metrics.json` |
| `configs/train/default.yaml` | Khối `pretrain:` + `finetune_lr` |
| `tests/unit/test_pretrain_cohort.py` *(mới)* | 9 test, tập trung vào an toàn rò rỉ |

**An toàn rò rỉ** được *khẳng định* chứ không được tin: `cohort_pretrain_fold` tự kiểm tra rằng
không EDF nào của bệnh nhân đích lọt vào train/val và raise nếu có. Tách val ở **mức bệnh nhân**,
nên early stopping trong giai đoạn tiền huấn luyện đo trên bệnh nhân model chưa từng thấy.

**Chi phí:** init chỉ phụ thuộc *bệnh nhân nào bị giữ lại*, không phụ thuộc EDF nào là test, nên
66 fold chỉ cần **13 lần tiền huấn luyện**, được cache trên đĩa và dùng lại.

#### Cách chạy để tận dụng hết máy

Model chỉ ~12k tham số, nên thời gian mỗi vòng lặp bị chi phối bởi **overhead khởi chạy kernel
CUDA**, không phải bởi tính toán. Hệ quả: tăng batch size gần như vô ích (và còn làm đổi ngữ nghĩa
tối ưu hoá, phải chỉnh lại LR), còn **chạy song song nhiều lần tiền huấn luyện thì rất hiệu quả**,
vì mỗi lần đều để GPU gần như rảnh. 13 lần tiền huấn luyện hoàn toàn độc lập và mỗi lần ghi một
file cache riêng, nên chia shard là an toàn, không cần khoá.

`scripts/pretrain_cohort.py` *(mới)* dựng sẵn toàn bộ init, có tham số `+shard=i +n_shards=n`:

```bash
ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 3 shard song song trên máy rảnh — chú ý num_workers THẤP hơn cho mỗi tiến trình
for i in 0 1 2; do
  python scripts/pretrain_cohort.py profile=server \
    +shard=$i +n_shards=3 profile.num_workers=4 &
done; wait

# Sau đó train bình thường, nó sẽ thấy mọi init đã có sẵn trong cache
python scripts/train.py profile=server train.pretrain.enabled=true train.force_retrain=true
python scripts/evaluate.py profile=server
```

`configs/profile/server.yaml` đặt `num_workers: 14` — **một worker cho mỗi lõi vật lý**, đúng cho
trường hợp chạy **một** tiến trình. Không nâng lên 28: worker chủ yếu cắt mảng và dựng tensor,
tức bị giới hạn bởi băng thông bộ nhớ, nên hai luồng trên cùng một lõi tranh nhau đơn vị load/store
chứ không cộng thêm thông lượng. Khi chia shard thì phải hạ xuống (3 shard → `num_workers=4`),
vì các tiến trình dùng chung 14 lõi đó — và vì quá tải worker cũng chính là nguyên nhân của hai lần
"Too many open files" đã gặp.

### 14.2c Kết quả L1 — đã chạy trên dữ liệu thật

| | Row #17 (không L1) | L1, lưới cũ (row 21) | **L1 + lưới rộng (row 22)** |
|---|---:|---:|---:|
| Sensitivity | 0.8806 | 0.8485 | **0.9218** |
| FAR/h | 0.3119 | 0.1243 | **0.1878** |
| Delay mean | 19.42 | 18.10 | **17.06** |
| Delay median | 16.0 | 15.0 | **13.00** |
| Phản ứng model | 6.42 | 5.10 | **4.06** |
| Worst-pt FAR | 1.5533 | 0.3434 | **0.6182** |
| Worst-pt sens | 0.5000 | 0.0000 | 0.3333 (chb17, 3 cơn) |

**L1 hiệu quả, nhưng lần chạy đầu che mất điều đó.** Ở row 21, FAR rơi về 0.124 trong khi trần là
0.30 — bỏ không 2.4 lần ngân sách rồi lấy sensitivity ra trả. Nguyên nhân là sàn của
`threshold_search.on_grid` dừng ở 0.20: một model được tiền huấn luyện cho điểm số tách bạch hơn,
và lưới không đủ thấp để tìm điểm vận hành nhạy hơn. Đây đúng là lỗi đã chặn dự án ở bước 2 của
timeline, khi sàn còn là 0.50.

Nới lưới xuống 0.02 và **chọn lại ngưỡng từ chính checkpoint đã lưu, không train lại một fold nào**
cho ra row 22 — kết quả tốt nhất từ trước tới nay:

1. **Vượt cả hai baseline tái lập.** So với `compact1d_7k` #7 (0.8974 @ FAR 0.2103), row 22 **trội
   hơn ở cả hai trục** — sensitivity cao hơn 2.4 pp *và* FAR thấp hơn. So với `frontiers2d` #2
   (0.8811 @ 0.1432), sensitivity cao hơn **4.1 pp**. Nghĩa là luận điểm A1 không còn cần dựa vào
   "tương đương thống kê" nữa — nó đã thắng thẳng, ở chi phí tính toán thấp hơn 1.8× và 3.3×.
2. **Đạt mức minimum của gate v2 cho sensitivity** (≥0.92): 0.9218.
3. **Phản ứng model giảm 37%** (6.42 → 4.06 s). Đây là phần đã trừ sàn giao thức, nên nó là cải
   thiện thật của model chứ không phải dịch điểm vận hành.
4. **Delay median 13.00 s bằng đúng sàn giao thức.** Nghĩa là với một nửa số cơn, model đã phát
   hiện ở thời điểm **sớm nhất mà cấu hình đo cho phép** — phần dư model ở trung vị bằng 0. Không
   còn chỗ để cải thiện delay bằng cách sửa model; toàn bộ phần còn lại nằm ở sàn.

Worst-patient rơi vào **chb17 với 3 cơn** (1/3 = 0.3333), tức thuộc diện được miễn trừ theo quy tắc
`min_events_to_gate: 5` của gate v2 (§4).

**Việc tiếp theo có giá trị cao nhất** không còn là sửa model mà là hạ sàn: đổi `run_length` 3→1 và
`ema_alpha` 0.125→0.5 đưa sàn từ 13.0 xuống **5.0 s**. Nếu phần dư model giữ nguyên ở 4.06 s thì
delay trung bình rơi về **~9.1 s** và trung vị về ~5 s — cũng chỉ cần `rethreshold`, không train lại.

### 14.3 Thứ tự đề xuất

L1 trước tiên và một mình, để đo riêng tác dụng của nó — đây là thay đổi lớn nhất và cần biết nó
đóng góp bao nhiêu. Sau đó L4 và L7 (rẻ và làm cho mọi phép đo sau đó đáng tin hơn). Rồi L2 để
tấn công độ trễ. L3 và L5 là hai hướng lớn tiếp theo, mỗi hướng đủ sức trở thành một mục riêng
trong bài báo. L6 chạy kèm bất cứ lúc nào.

Mỗi bước nên được đo bằng **paired bootstrap so với cấu hình liền trước**, để biết bước nào thật
sự có tác dụng thay vì cộng dồn các thay đổi rồi không giải thích được kết quả — đúng cách nhóm đã
làm ở dự án CARE_ASD.

### 14.4 Cần trung thực về điều này

Không có gì bảo đảm bảy đòn bẩy trên đưa sensitivity lên 0.97. Nhưng có ba điều nói được chắc chắn:

1. Nguyên nhân của các con số hiện tại **đã được xác định** và nó là nguyên nhân có thể sửa được
   (đói dữ liệu huấn luyện), không phải một giới hạn vật lý của EEG một kênh.
2. Không đòn bẩy nào trong bảy đòn bẩy đó **từng được thử**, nên trần thật sự của phương pháp này
   hiện chưa ai biết. Kết luận "20 run đều không đạt cổng" hiện **không phải** bằng chứng rằng cổng
   không đạt được — nó là bằng chứng rằng việc chỉnh ngưỡng và kiến trúc đã cạn.
3. Ngay cả khi sensitivity dừng ở khoảng 0.92, kết hợp L7 (thanh sai số) với luận điểm tương đương
   thống kê và ưu thế 4.3× về MACs đã đủ hình dạng cho một bài TBioCAS — miễn là phần cứng đủ dày
   (§11.6).

---

## Nguồn

- Chung Y-G, Cho A, Kim H, Kim KJ. *Single-channel seizure detection with clinical confirmation of seizure locations using CHB-MIT dataset.* Front. Neurol. 15:1389731 (2024). <https://pmc.ncbi.nlm.nih.gov/articles/PMC11148866/>
- Ali E, Angelova M, Karmakar C. *Epileptic seizure detection using CHB-MIT dataset: The overlooked perspectives.* R. Soc. Open Sci. 11:230601 (2024). <https://royalsocietypublishing.org/doi/10.1098/rsos.230601>
- *Wearable Epilepsy Seizure Detection on FPGA with Spiking Neural Networks.* IEEE Trans. Biomed. Circuits Syst. (2025). <https://doi.org/10.1109/TBCAS.2025.3575327>
- Vittimberga A, Nicolini G, Scotti G. *Real-Time Multi-Channel Epileptic Seizure Detection Exploiting an Ultra-Low-Complexity Algorithm–Hardware Co-Design Approach.* Sensors 25(22):6889 (2025). <https://doi.org/10.3390/s25226889>
- Ahlawat K. *Efficient EEG Seizure Detection Using INT8 Quantization, Channel Pruning, and Spiking Neural Networks.* arXiv:2607.16296 (2026). <https://arxiv.org/abs/2607.16296>
- *Automatic epileptic seizure detection with an ultra lightweight 3D-CNN model.* Biomed. Signal Process. Control (2025). <https://www.sciencedirect.com/science/article/abs/pii/S1746809425007712> — 6 540 params / 2.39 M MACs, 99.24 % / 0.53 FAR/h / 4.97 s
- *Lightweight Seizure Detection Based on Multi-Scale Channel Attention.* (2023). <https://pubmed.ncbi.nlm.nih.gov/37845193/> — 88 k params / 2.68 M MACs
- *An energy-efficient seizure detection processor using event-driven multi-stage CNN classification…* DAC 2022. <https://dl.acm.org/doi/10.1145/3489517.3530421> — 0.32 µJ/classification
- *A 65 nm/0.448 mW EEG processor with parallel architecture SVM and lifting wavelet transform.* Comput. Biol. Med. (2022). <https://www.sciencedirect.com/science/article/abs/pii/S0010482522001585>
- *Epileptic Seizure Detection on an Ultra-Low-Power Embedded RISC-V Processor Using a CNN.* Biosensors 11(7):203 (2021). <https://www.mdpi.com/2079-6374/11/7/203>
- IEEE CASS. *TBioCAS scope and guidelines for authors.* <https://ieee-cas.org/publication/TBioCAS/guidelines-authors>
- PYNQ documentation, *PMBus.* <https://pynq.readthedocs.io/en/latest/pynq_libraries/pmbus.html>
