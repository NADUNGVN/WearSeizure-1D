# Kế hoạch: chọn định dạng số bằng đo, không bằng lập luận

Câu hỏi cần trả lời: **dynamic fixed point có thật sự tốt hơn INT8 không, và
trong trường hợp nào thì chọn cái nào.**

Đây là mục **A4** — mất mát do lượng tử hoá — thứ duy nhất còn chặn con số độ
chính xác cuối cùng, và là mục dự án **chưa từng đo một lần nào**.

---

## 1. "Dynamic fixed point thay cho INT8" gộp hai thay đổi ngược chiều

Đây là điều phải tách ra trước khi đo, nếu không kết quả sẽ không diễn giải được.

| thay đổi | tác động tới độ chính xác | tác động tới phần cứng |
|---|---|---|
| **nới độ rộng bit** 8 → 16 | **tốt hơn** | gấp đôi bộ nhớ; **không tốn thêm DSP** trên Zynq-7020 |
| **ép scale về luỹ thừa 2** | **kém hơn** — scale làm tròn lên, mất tới 2× dải | bỏ được một **bộ nhân** ở mỗi ranh giới layer; requantise chỉ còn dịch bit |

Đo được ở cùng 8 bit trên dữ liệu ngẫu nhiên: sai số tuyệt đối trung bình
**0.031** với scale luỹ thừa 2 so với **0.025** với scale số thực — kém hơn 25 %.

Nên nếu chỉ so "DFP16 vs INT8" rồi thấy DFP16 thắng, ta **không biết** phần
thắng đến từ đâu. Lưới thí nghiệm dưới đây tách hai trục.

---

## 2. Lưới thí nghiệm — 2 trục, 6 ô

|  | scale số thực | **scale luỹ thừa 2** (fixed point) |
|---|---|---|
| **16 bit** | `int16` | **`dfp16`** |
| **8 bit** | `int8_ptq` | **`dfp8`** |
| **4b W / 8b A** | `w4a8_qat` | — |

Cộng `fp32` làm mốc. Tất cả config đã có trong `configs/precision/`.

Đọc theo hàng và theo cột:

- **`int16` → `dfp16`** và **`int8_ptq` → `dfp8`**: giá phải trả của ràng buộc
  luỹ thừa 2, đo hai lần ở hai độ rộng.
- **`dfp16` → `dfp8`** và **`int16` → `int8_ptq`**: giá phải trả của việc hẹp bit.
- Nếu `dfp16` ≈ `fp32` mà `dfp8` mất nhiều, kết luận là **độ rộng bit** quyết
  định, không phải kiểu scale.

## 3. Cách đo

**Post-training quantization, không train lại.** Dùng đúng 66 fold × 3 seed
checkpoint đã có của `k5only` + L1 + L8. Với mỗi ô:

1. Hiệu chuẩn scale trên **tập validation của từng fold** — không phải test.
   Đây là quy tắc của cả dự án và không được nới ở đây.
2. Chạy inference lượng tử hoá trên tập test, giữ nguyên hậu xử lý và ngưỡng
   **đã đóng băng từ FP32** — nếu chỉnh lại ngưỡng cho từng định dạng thì phép
   đo trở thành "định dạng nào chịu chỉnh ngưỡng tốt hơn", không phải "định dạng
   nào mất bao nhiêu".
3. Báo cáo **Δ so với FP32** theo pp cho: sensitivity mức sự kiện, FAR/h,
   worst-patient sensitivity.
4. Paired bootstrap theo cụm bệnh nhân giữa mỗi định dạng và FP32.

Chi phí: không train, chỉ inference. 66 × 3 × 6 ô ≈ vài giờ trên một máy.

## 4. Tiêu chí chọn — quyết định trước khi thấy số liệu

Ngưỡng lấy từ `configs/precision/int8_qat.yaml`, không phải đặt mới:

| | ngưỡng |
|---|---|
| mục tiêu | Δ ≤ **0.5 pp** so với FP32 |
| tối thiểu chấp nhận | Δ ≤ **1.0 pp** |

**Chọn định dạng RẺ NHẤT còn nằm trong 0.5 pp.** Thứ tự rẻ dần:
`dfp8` < `int8_ptq` < `dfp16` < `int16` < `fp32`.

Nếu không định dạng nào trong 0.5 pp: nới lên 1.0 pp và **ghi rõ trong bài** là
đã nới, kèm lý do.

Nếu vẫn không: chuyển sang **QAT** (`int8_qat`, `w4a8_qat`) — huấn luyện có mô
phỏng lượng tử hoá thường lấy lại được phần lớn mất mát mà PTQ để rơi.

## 5. Vì sao mong đợi mạng nhỏ mất nhiều hơn

11 786 tham số là rất nhỏ. Mạng nhỏ có **ít dư thừa để hấp thụ nhiễu lượng tử
hoá**, nên PTQ thường tổn hại nhiều hơn so với mạng lớn.

Hai bằng chứng trực tiếp, cả hai đều nằm trong bảng so sánh của dự án:

**1. Lee et al. 2024, IEEE TBioCAS — đây là bằng chứng mạnh nhất.** Cùng bài
toán phát hiện động kinh, **cùng board Xilinx PYNQ-Z2** mà dự án này nhắm tới,
đăng cùng tạp chí đích. Họ chọn **fixed point 16 bit**, và trích nguyên văn:

> *"quantizing the 32-bit floating-point numbers computed in software into
> 16-bit fixed-point numbers on hardware not only saves 50% of memory space but
> also maintains an accuracy rate of 99.3%"*

FP32 của họ là 99.5 %, nên **16-bit fixed mất đúng 0.2 pp** — nằm gọn trong
ngưỡng 0.5 pp của dự án. Họ **không** thử 8 bit.

Thiết kế của họ dùng **7 khối BRAM** trên PYNQ-Z2, tức khoảng 5 % của 140 khối.
Con số này xác nhận ước lượng ~5 khối cho bản INT8 và ~10 khối cho bản 16 bit ở
§6 là đúng thang.

Lưu ý khi trích dẫn: model của họ chỉ **356 tham số** vì mỗi lớp conv chỉ có
**một kernel** `(k=6, stride=2, 1 kernel)`, và một khối **DWT 2 mức** làm phần
lớn việc trích đặc trưng trước khi vào CNN. Dữ liệu là **EEG chuột**, không phải
CHB-MIT. Nên đây là bằng chứng về **lựa chọn định dạng số trên đúng thiết bị**,
không phải điểm so sánh về độ khó bài toán.

**2. EpiSepNet-5K — đây là model trước của chính nhóm này.** 5 010 tham số trên
cùng CHB-MIT, và nó cũng đã chọn **INT16 chứ không phải INT8**, báo **99.9743 %
khớp với FP32** và gói nhỏ hơn **2.81×**.

Điều này quan trọng hơn một dẫn chứng từ nhóm khác: đó là **tiền lệ của chính
nhóm**. Nếu lần này chọn INT8 thì phải giải thích được vì sao đi ngược lại lựa
chọn trước của mình trên một model **còn nhỏ hơn nữa** — 11 786 tham số nhưng
chỉ **một kênh** thay vì 17, tức mỗi tham số phải gánh nhiều hơn.

## 6. Vì sao bit rộng hơn gần như miễn phí ở đây

Hai ràng buộc thông thường ép người ta xuống 8 bit, và **không cái nào áp dụng**:

**Bộ nhớ.** Sinh bằng `python scripts/hardware_spec.py wearseizure1d_k5only`:

| định dạng | trọng số | kích hoạt (streaming) | **tổng SRAM** | % BRAM XC7Z020 |
|---|---:|---:|---:|---:|
| FP32 | 46.0 KiB | 26.7 KiB | **72.7 KiB** | ~12 % |
| INT16 / DFP16 | 23.0 KiB | 13.3 KiB | **36.3 KiB** | ~6 % |
| INT8 / DFP8 | 11.5 KiB | 6.7 KiB | **18.2 KiB** | ~3 % |

Ngay cả **FP32 chưa nén** cũng vừa. Bộ nhớ không phải thứ quyết định.

**Bộ nhân.** DSP48E1 của Zynq-7020 là bộ nhân **25×18**, nên một phép nhân
16×16 **vừa đúng một DSP** — cùng một DSP mà phép 8×8 dùng. Nới lên 16 bit
**không tốn thêm DSP**, trừ khi nhồi hai phép INT8 vào một DSP, cách chỉ đáng
khi DSP khan hiếm. Thiết kế này cần **4–8 MAC**, board có **220 DSP**.

> Nhiều người mặc định INT8 rẻ hơn INT16 về phần cứng. Trên thiết bị này, với
> số MAC nhỏ thế này, **không**.

## 6b. Footprint của dynamic fixed point

**Footprint dữ liệu của DFP giống hệt số nguyên cùng độ rộng** — DFP16 = INT16 =
36.3 KiB, DFP8 = INT8 = 18.2 KiB. Scale là siêu dữ liệu **mỗi tensor** (hoặc mỗi
kênh), không phải mỗi giá trị, nên nó không nhân lên theo số trọng số.

Chỗ **duy nhất** DFP nhỏ hơn là cách lưu scale: scale số thực cần một `float32`,
scale luỹ thừa 2 chỉ cần **số mũ nguyên**, 1 byte.

| độ mịn | số scale | scale float32 | số mũ luỹ thừa 2 | tiết kiệm |
|---|---:|---:|---:|---:|
| mỗi tensor | 29 | 116 B | 29 B | 87 B |
| **mỗi kênh** | 465 | 1 860 B | 465 B | **1 395 B** |

Ở mức mỗi tensor thì không đáng kể. Ở mức **mỗi kênh** — độ mịn mà lượng tử hoá
hiện đại thường dùng vì nó chính xác hơn hẳn — DFP tiết kiệm **1.4 KiB, tức 12 %
bộ nhớ trọng số INT8**. Không lớn, nhưng không phải không có.

Kết luận cho việc chọn: **đừng chọn DFP vì footprint.** Chênh lệch là 87 B tới
1.4 KiB trên tổng 18.2 KiB, và bộ nhớ vốn không phải ràng buộc. Chọn nó vì §7.

## 7. Vậy lợi ích thật của dynamic fixed point là gì

Không phải bộ nhớ, không phải DSP. Là **bộ nhân bị bỏ khỏi đường requantise**.

Giữa hai layer, một đường INT8 với scale số thực phải làm `nhân + dịch` cho mỗi
phần tử đầu ra. Với scale luỹ thừa 2 thì chỉ còn `dịch`. Với 13 layer, đó là 13
bộ nhân requantise biến mất khỏi datapath.

Đó là lý do nên chọn nó — và nó **độc lập** với việc chọn 8 hay 16 bit. Hai
quyết định tách rời, và lưới ở §2 đo chúng tách rời.

## 8. Việc cần làm

| # | việc | trạng thái |
|---|---|---|
| 1 | Scale luỹ thừa 2 trong `QuantScale` | **xong** (`8d19c85`) |
| 2 | Config `dfp8`, `dfp16` | **xong** |
| 3 | Đường PTQ đọc `power_of_two_scale` từ config | chưa |
| 4 | Script quét: 6 ô × 66 fold × 3 seed, báo Δ so với FP32 | chưa |
| 5 | Chạy, ghi vào `EXPERIMENT_LOG_G1a.md` | chưa |
| 6 | Chốt định dạng, cập nhật `MODEL_CARD` và `HARDWARE_HANDOFF` | chưa |

Việc 3 và 4 làm được ngay tại máy local; việc 5 cần checkpoint trên server.
