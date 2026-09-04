# Chọn phương pháp Post-Training Quantization

Bước tiếp theo là so **DFP với INT**. Nhưng phép so đó chỉ có nghĩa nếu **phương
pháp PTQ đủ tốt** — nếu calibration kém, cả hai định dạng cùng hỏng và ta sẽ kết
luận sai rằng "8 bit không dùng được".

Tài liệu này chọn phương pháp PTQ trước, rồi mới chạy lưới định dạng.

---

## 1. Vì sao kiến trúc này là trường hợp xấu nhất cho PTQ ngây thơ

`wearseizure1d_k5only` là **depthwise-separable từ đầu đến cuối** — `b1`,
`b2/b3/b4`, và cả hai lớp `context` đều là depthwise rồi pointwise.

Đây **chính xác** là họ kiến trúc mà PTQ per-tensor được biết là hỏng nặng.
Krishnamoorthi (2018), *Quantizing deep convolutional networks for efficient
inference: A whitepaper*, cho thấy MobileNet — cũng depthwise-separable — mất
rất nhiều độ chính xác với INT8 per-tensor, trong khi **per-channel thì gần như
không mất gì**.

Nguyên nhân: một lớp depthwise có **một filter độc lập cho mỗi kênh**, không có
phép trộn kênh nào để cân bằng chúng. Các kênh vì thế trôi về những dải giá trị
rất khác nhau. Một scale chung cho cả tensor bị kênh có biên độ lớn nhất chi
phối, và mọi kênh biên độ nhỏ bị nén xuống còn vài mức lượng tử.

Model này còn ở thế bất lợi hơn MobileNet: **11 786 tham số**, gần như không có
dư thừa để hấp thụ nhiễu.

---

## 2. Phép đo phải làm TRƯỚC mọi thứ khác

**Độ tản biên độ theo kênh trên trọng số ĐÃ TRAIN.**

Đo trên trọng số khởi tạo ngẫu nhiên cho tỉ lệ max/min chỉ **1.0–2.6×** — nhưng
con số đó vô nghĩa, vì khởi tạo ngẫu nhiên vốn đồng đều. Điều cần biết là các
kênh **trôi ra bao xa sau khi train**.

```bash
# trên SERVER-02, dùng checkpoint L1+L8 đã có
python scripts/measure_weight_ranges.py profile=server data=chbmit \
    model=wearseizure1d_k5only train.run_tag=L8
```

Cách đọc, và nó **quyết định toàn bộ phần còn lại**:

| tỉ lệ max/min theo kênh | nghĩa là | làm gì |
|---|---|---|
| **< 4×** | các kênh đồng đều | per-tensor có thể đủ; bỏ qua CLE, tiết kiệm nhiều công |
| **4–20×** | tản vừa | **per-channel là bắt buộc**; CLE có thể giúp thêm |
| **> 20×** | tản nặng, đúng bệnh MobileNet | per-channel + **CLE + bias correction** |

Không đo cái này mà đi cài CLE ngay là làm thừa; mà bỏ qua nó rồi kết luận "8
bit không dùng được" thì lại là kết luận sai.

### ĐÃ ĐO (04-09) — kết quả: UNIFORM, 3.4×

198 checkpoint, 3 seed, tag `L8`, trên SERVER-02:

| layer | kiểu | out_ch | trung vị | max |
|---|---|--:|--:|--:|
| `stem.0` | conv | 8 | 1.8× | 4.4× |
| `b1.depthwise` | depthwise | 8 | 1.7× | 2.2× |
| `b2.branch_k5` | depthwise | 16 | 1.9× | 2.8× |
| `b3.branch_k5` | depthwise | 24 | 2.0× | 3.4× |
| `b4.branch_k5` | depthwise | 32 | 2.4× | 3.8× |
| `context.0.depthwise` | depthwise | 48 | 2.2× | 3.7× |
| **`context.1.depthwise`** | depthwise | 64 | **3.4×** | **7.4×** |
| các pointwise | conv | 16–64 | 1.9–2.2× | 2.6–3.1× |

**Tản tối đa 3.4× — dưới ngưỡng 4×. Bệnh MobileNet KHÔNG xảy ra ở model này.**

Hệ quả trực tiếp, và nó cắt bỏ phần đắt nhất của kế hoạch:

- **Bỏ P4 (Cross-Layer Equalisation).** Xây CLE cho các kênh đã đều thế này là
  làm thừa. Đây là lý do phép đo này phải chạy trước, và nó vừa tiết kiệm phần
  công lớn nhất trong toàn bộ thang.
- **Bỏ P2 (bias correction)** khỏi đường chính — nó tồn tại chủ yếu để cứu các
  trường hợp lệch nặng.
- **P1 (per-channel) giữ lại như bảo hiểm rẻ tiền**, không phải như thứ bắt
  buộc. Ở 3.4×, per-tensor mất khoảng `log2(3.4) ≈ 1.8` bit trên kênh nhỏ nhất —
  còn ~6 bit ở INT8, đủ dùng. Nhưng cột `max` cho thấy vài fold lên tới **7.4×**,
  và per-channel đã cài xong nên chi phí giữ nó bằng không.

Một chi tiết đáng chú ý: layer tản nhiều nhất là **`context.1.depthwise`** —
đúng layer k5 dilation 16 đọc phần lớn padding. Các kênh ở đó nhận tín hiệu
gradient kém nhất quán nhất, nên trôi xa nhau nhất. Điều này khớp với mọi thứ
khác đã biết về layer đó.

---

## 3. Các phương pháp PTQ, xếp theo mức phù hợp với model này

### 3.1 Per-channel weight quantization — đòn bẩy lớn nhất

Một scale riêng cho **mỗi kênh đầu ra** thay vì một scale cho cả tensor. Với
mạng depthwise-separable đây là khác biệt giữa "dùng được" và "hỏng".

Chi phí: **465 scale thay vì 29**. Ở DFP (số mũ 1 byte) là **465 B**; ở scale
float32 là 1 860 B. Đã tính ở `PLAN_quantisation.md` §6b.

Trạng thái trong repo: **chưa có.** `compute_symmetric_scale` lấy
`x.abs().max()` trên toàn tensor. Sửa thành per-channel là thay đổi nhỏ.

> **Lưu ý phần cứng:** per-channel scale nghĩa là mỗi kênh đầu ra có một hệ số
> requantise riêng. Với DFP đó là **một shift riêng mỗi kênh** — vẫn chỉ là
> shift, không phải nhân. Đây là điểm per-channel và DFP hợp nhau.

### 3.2 Cross-Layer Equalization + bias correction

Nagel và cộng sự, *Data-Free Quantization Through Weight Equalization and Bias
Correction*, ICCV 2019.

CLE khai thác tính **đẳng biến với phép co giãn dương của ReLU**: nhân trọng số
lớp `i` với `s` theo kênh và chia trọng số lớp `i+1` cho `s` cho ra **đúng cùng
một hàm**, nhưng biên độ các kênh được san phẳng. Chọn `s` sao cho dải giá trị
đều nhau thì per-tensor quantisation trở nên khả thi.

Nó được thiết kế **đúng cho mạng depthwise-separable**, và cặp
`depthwise → pointwise` trong model này là cặp tự nhiên để cân bằng.

Bias correction sửa độ lệch trung bình mà lượng tử hoá gây ra — gần như miễn phí
và thường lấy lại được vài phần mười điểm.

**Cả hai đều data-free** — không cần dữ liệu hiệu chuẩn, nên không có rủi ro rò
rỉ nào.

Điều kiện áp dụng ở model này: mọi conv đều theo sau bởi BN + ReLU, nên tính
đẳng biến giữ được. **Trừ `context.1.pointwise`**, lớp cuối trước GAP — cần kiểm
tra hàm kích hoạt ở đó trước khi cân bằng qua nó.

### 3.3 Hiệu chuẩn kích hoạt tốt hơn absmax

Hiện tại `_quantize_activation` dùng **absmax chạy trung bình động** với
`momentum=0.1`. Đây là lựa chọn nhạy với ngoại lai: một cửa sổ có nhiễu chuyển
động sẽ kéo scale lên và làm mọi giá trị bình thường mất phân giải.

EEG **có** ngoại lai như vậy — nhiễu điện cực, cử động, chớp mắt.

Ba lựa chọn thay thế, xếp theo độ phức tạp:

| cách | làm gì | ghi chú |
|---|---|---|
| **percentile** | cắt ở 99.9% thay vì max | một dòng code, thường đủ |
| **MSE-optimal** | quét ngưỡng cắt, chọn cái tối thiểu hoá sai số bình phương | khoẻ hơn, vẫn rẻ |
| **entropy / KL** | tối thiểu hoá KL giữa phân phối FP và lượng tử | cách TensorRT dùng |

Khuyến nghị **MSE-optimal**: rẻ, không có siêu tham số phải chỉnh (percentile
phải chọn 99.9 hay 99.99), và là mặc định trong hầu hết thư viện hiện đại.

### 3.4 AdaRound — chỉ khi cần

Nagel và cộng sự, *Up or Down? Adaptive Rounding for Post-Training
Quantization*, ICML 2020.

Thay vì làm tròn tới gần nhất, **học** hướng làm tròn cho từng trọng số bằng một
bài tối ưu nhỏ trên tập hiệu chuẩn, không cần nhãn. Lợi ích lớn ở bit thấp
(≤4 bit), vừa phải ở 8 bit.

Đây là bước cuối, chỉ chạm tới nếu 3.1–3.3 không đủ. Nó phức tạp hơn hẳn và nếu
đã phải tới đây thì **QAT có thể là lựa chọn đúng hơn** — repo đã có sẵn QAT.

---

## 4. Thang thí nghiệm

Chạy **tuần tự**, dừng ngay khi đạt ngưỡng 0.5 pp. Tất cả ở **INT8**, vì đó là
trường hợp khó nhất — cái gì sống được ở INT8 thì sống được ở INT16.

| bậc | thêm gì | công phải làm |
|---|---|---|
| **P0** | per-tensor absmax (hiện tại) | không, đã có |
| **P1** | + **per-channel weights** | **đã cài** (`991bc82`) |
| ~~P2~~ | ~~bias correction~~ | **bỏ** — §2 đo được kênh đều |
| **P3** | + **hiệu chuẩn kích hoạt MSE** | vừa — **đây là bậc còn lại đáng làm** |
| ~~P4~~ | ~~CLE~~ | **bỏ** — §2 đo được 3.4× < 4× |
| **P5** | AdaRound *hoặc* chuyển sang QAT | lớn, chỉ nếu P3 không đủ |

Sau phép đo ở §2, thang rút xuống còn **P0 → P1 → P3**. Trọng số đã đều, nên
rủi ro còn lại nằm ở **kích hoạt**, không phải trọng số — và EEG có ngoại lai
thật (nhiễu điện cực, cử động), đúng thứ mà absmax xử lý tệ nhất.

**Chỉ khi đã chốt được bậc PTQ** mới chạy lưới định dạng ở
`PLAN_quantisation.md` §2 — bốn ô `{8, 16 bit} × {scale thực, luỹ thừa 2}`.

Làm ngược lại sẽ đo nhầm: nếu PTQ ở bậc P0 thì INT8 hỏng vì **calibration**, và
ta sẽ ghi vào log rằng "8 bit không dùng được" trong khi thật ra là "cách hiệu
chuẩn của chúng ta không dùng được".

---

## 5. Quy tắc không được phá

**Hiệu chuẩn trên tập validation của từng fold, không phải test.** Đây là quy
tắc của cả dự án. Một PTQ hiệu chuẩn trên test là một dạng rò rỉ khác, và cả bài
báo này tồn tại để chỉ ra chuyện đó ở nơi khác.

**Ngưỡng hậu xử lý đóng băng từ FP32.** Nếu chỉnh lại ngưỡng cho từng định dạng
thì phép đo trở thành "định dạng nào chịu chỉnh ngưỡng tốt hơn", không phải
"định dạng nào mất bao nhiêu".

**Báo cáo Δ so với FP32 theo pp**, kèm paired bootstrap theo cụm bệnh nhân —
giống mọi so sánh khác trong dự án. Trên 13 bệnh nhân, chênh lệch dưới ~3 pp
không phân giải được, nên một "mất 0.4 pp" cần khoảng tin cậy đi kèm mới có
nghĩa.

---

## 6. Nguồn

- R. Krishnamoorthi. *Quantizing deep convolutional networks for efficient
  inference: A whitepaper.* arXiv:1806.08342, 2018. — per-tensor so với
  per-channel trên mạng depthwise-separable.
- M. Nagel, M. van Baalen, T. Blankevoort, M. Welling. *Data-Free Quantization
  Through Weight Equalization and Bias Correction.* ICCV 2019. — CLE và bias
  correction.
- M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, T. Blankevoort. *Up or Down?
  Adaptive Rounding for Post-Training Quantization.* ICML 2020. — AdaRound.
- M. Nagel và cộng sự. *A White Paper on Neural Network Quantization.*
  arXiv:2106.08295, 2021. — tổng quan PTQ, và là tài liệu nên đọc trước khi cài.
