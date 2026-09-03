# Bàn giao model cho nhóm thiết kế phần cứng

Trạng thái: **kiến trúc ĐÃ CHỐT là `wearseizure1d_k5only`** — xem §4. Tài liệu này ghi lại con số tốt nhất
hiện có và mọi thứ nhóm accelerator cần, để phần việc không phụ thuộc lựa chọn
cuối có thể bắt đầu ngay.

Sinh lại mọi bảng per-layer bằng:

    python scripts/hardware_spec.py <model_name> [--markdown]

Không con số nào dưới đây chép tay; tất cả đọc trực tiếp từ model bằng forward hook.

---

## 1. Điều quan trọng nhất: "recipe" khác "kiến trúc"

Ba đòn bẩy đã đo — L4 (chọn checkpoint theo AUPRC), L8 (chưng cất từ
`frontiers2d`), L3 (chưng cất đa kênh) — **không đổi mạng**. Chúng đổi *giá trị
trọng số*, không đổi layer, không đổi MACs, không đổi line buffer.

Hệ quả trực tiếp cho tiến độ: **RTL có thể bắt đầu với một kiến trúc đã chốt, và
bộ trọng số tốt nhất nạp vào sau.** Đổi recipe không tốn một dòng RTL nào.

Thứ **duy nhất** buộc phải làm lại RTL là thang dung lượng (`ctx16` / `wide`), vì
nó đổi số kênh từng tầng.

## 2. Kiến trúc ứng viên và dấu chân phần cứng

| | `k5only` (hiện tại) | `k5only_ctx16` | `k5only_wide` |
|---|---:|---:|---:|
| context channels | 64 | 16 | 16 |
| stage channels | 16/24/32/48 | 16/24/32/48 | 24/36/48/72 |
| MACs conv+fc (triển khai) | 489 600 | **285 216** | 513 568 |
| MACs thop (số dùng cho cổng) | 585 920 | **367 664** | 626 736 |
| tham số = bộ nhớ trọng số INT8 | 11 786 B (11.5 KiB) | **5 114 B (5.0 KiB)** | 9 414 B (9.2 KiB) |
| tổng line buffer INT8 | 6 823 B (6.7 KiB) | **3 655 B (3.6 KiB)** | 4 927 B (4.8 KiB) |
| **tổng SRAM on-chip** | **18.2 KiB** | **8.6 KiB** | **14.0 KiB** |
| buffer lớn nhất | `context.1.depthwise` 4 160 (61%) | `context.0.depthwise` 1 584 (43%) | `context.0.depthwise` 2 376 (48%) |

Hai quy ước MAC lệch nhau vì `thop` tính cả BatchNorm và phép elementwise.
**BatchNorm gấp được vào conv đứng trước khi suy luận**, nên số accelerator thật
sự phải phát ra là cột conv+fc. Cổng MAC và mọi so sánh giữa các model trong bài
dùng số `thop`, nhất quán cho mọi model.

## 3. Bảng per-layer — `wearseizure1d_k5only`

Đầu vào: **1 kênh × 1024 mẫu** (4 s @ 256 Hz), một cửa sổ trượt mỗi 1 s.

| layer | kiểu | in | out | k | s | dil | out_len | taps | buffer | MACs | trọng số |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `stem.0` | conv | 1 | 8 | 7 | 2 | 1 | 512 | 7 | 7 | 28 672 | 56 |
| `b1.depthwise` | depthwise | 8 | 8 | 5 | 2 | 1 | 256 | 5 | 40 | 10 240 | 40 |
| `b1.pointwise` | conv | 8 | 16 | 1 | 1 | 1 | 256 | 1 | 8 | 32 768 | 128 |
| `b2.branch_k5` | depthwise | 16 | 16 | 5 | 2 | 1 | 128 | 5 | 80 | 10 240 | 80 |
| `b2.pointwise` | conv | 16 | 24 | 1 | 1 | 1 | 128 | 1 | 16 | 49 152 | 384 |
| `b3.branch_k5` | depthwise | 24 | 24 | 5 | 2 | 2 | 64 | 9 | 216 | 7 680 | 120 |
| `b3.pointwise` | conv | 24 | 32 | 1 | 1 | 1 | 64 | 1 | 24 | 49 152 | 768 |
| `b4.branch_k5` | depthwise | 32 | 32 | 5 | 2 | 4 | 32 | 17 | 544 | 5 120 | 160 |
| `b4.pointwise` | conv | 32 | 48 | 1 | 1 | 1 | 32 | 1 | 32 | 49 152 | 1 536 |
| `context.0.depthwise` | depthwise | 48 | 48 | 5 | 1 | 8 | 32 | 33 | 1 584 | 7 680 | 240 |
| `context.0.pointwise` | conv | 48 | 64 | 1 | 1 | 1 | 32 | 1 | 48 | 98 304 | 3 072 |
| `context.1.depthwise` | depthwise | 64 | 64 | 5 | 1 | 16 | 32 | 65 | 4 160 | 10 240 | 320 |
| `context.1.pointwise` | conv | 64 | 64 | 1 | 1 | 1 | 32 | 1 | 64 | 131 072 | 4 096 |

Sau đó: GAP trên chiều thời gian (32 → 1) → `Linear(64, 2)`.

Line buffer của một conv 1D chạy streaming là `(k−1)·dilation + 1` mẫu đầu vào
× `in_channels`. **Dilation, không phải kernel size, mới là thứ làm layer đắt bộ
nhớ.**

### Chi tiết nên biết trước khi sizing

`context.1.depthwise` chiếm **61% toàn bộ line buffer** vì k5 ở dilation 16 trải
**65 mẫu trên một chuỗi chỉ dài 32** — phần lớn tap đọc padding. Đây là layer
kém hiệu quả nhất thiết kế, và là lý do tồn tại nhánh `ctx16`.

Cảnh báo: **cắt nó không miễn phí.** Đo trên 3 seed, `ctx16` cho sensitivity
macro thấp hơn 2.33pp (CI [−6.60, +0.60], không có ý nghĩa thống kê, nhưng cohort
13 bệnh nhân không đủ mạnh để phân giải chênh lệch cỡ đó). Xem
`EXPERIMENT_LOG_G1a.md` §2h.

## 4. Kiến trúc đã CHỐT: `wearseizure1d_k5only`

Thang dung lượng đã chạy xong (rows 44, 52). Cả hai phương án thay thế đều
**không** tốt hơn, nên không còn gì có thể buộc làm lại RTL.

| | sens macro | sens micro | FAR/h | MACs | params |
|---|---:|---:|---:|---:|---:|
| **`k5only`** (chốt) | **0.9358** | 0.9351 | 0.2216 | 585 920 | 11 786 |
| `k5only_ctx16` | 0.9126 | 0.9221 | 0.2378 | 367 664 | 5 114 |
| `k5only_wide` | 0.9245 | 0.9351 | **0.2180** | 626 736 | **9 414** |

- `ctx16` (cắt context 64→16) **mất 2.33pp** sensitivity macro. Không lấy được
  37% MACs miễn phí.
- `wide` (ctx16 + stage rộng 50%) **ngang control**: Δ macro −1.14pp, CI
  [−3.04, +0.51]; micro **trùng tới 16 chữ số thập phân** — đúng 72/77 cơn.

`wide` vẫn có một luận điểm Pareto: ngang chất lượng với **20% ít tham số hơn**,
FAR và delay nhỉnh hơn chút. Nếu nhóm phần cứng muốn tiết kiệm bộ nhớ trọng số
(9.2 KiB so với 11.5 KiB) thì đó là lựa chọn hợp lệ. Nhưng nó **không** tốt hơn
về chất lượng, nên mặc định là `k5only`.

---

## 5. Chất lượng tốt nhất hiện có

Kiến trúc `k5only`, 3 seed, 66 fold, `patient_specific_loso_edf`, 185.0 h
exposure. **Ba dòng này khác nhau ở TRỌNG SỐ, không ở mạng** — cùng layer, cùng
MACs, cùng line buffer, nên lựa chọn giữa chúng **không ảnh hưởng phần cứng** và
có thể hoãn tới sau khi có RTL.

| recipe | sens macro | sens micro | FAR/h | delay | worst-pt sens | worst-pt FAR |
|---|---:|---:|---:|---:|---:|---:|
| L1 (row 32) | 0.9358 | 0.9351 | 0.2216 | 18.83 | 0.8714 | 0.7785 |
| L1 + L4 (row 39) | 0.9380 | — | **0.1904** | 18.83 | — | **0.5701** |
| **L1 + L8 (row 43)** | **0.9489** | **0.9567** | 0.2937 | **17.75** | **0.9333** | 2.1065 |

**Tốt nhất về sensitivity: L1 + L8.** Tốt nhất về FAR: L1 + L4. Không recipe nào
trội toàn diện.

### Đối chiếu mốc

| mốc | ngưỡng | L1+L8 | |
|---|---|---:|---|
| M2 sensitivity macro | ≥ 0.95 | 0.9489 | ❌ hụt 0.11pp |
| M2 sensitivity micro | ≥ 0.95 | 0.9567 | ✅ |
| M3 FAR/h | ≤ 0.20 | 0.2937 | ❌ (L4 đạt 0.1904) |
| M4 worst-patient sens | ≥ 0.85 | 0.9333 | ✅ |
| M5 worst-patient FAR | ≤ 0.50 | 2.1065 | ❌ |
| M8 exposure | ≥ 185 h | 185.0 | ✅ |
| M9 thanh sai số | 3 seed | 3 | ✅ |
| M10 MACs | ≤ 630 832 | 585 920 | ✅ |

Khoảng cách macro tới 0.95 là **0.11pp** — nhỏ hơn ảnh hưởng của một cơn duy
nhất trên 77 (1.28pp), tức nằm trong nhiễu seed. Micro đã vượt.

### Vì sao L1+L8 xấu ở worst-patient FAR

L8 chưng cất từ `frontiers2d`, và student học luôn **chế độ hỏng** của teacher:
chb23 FAR đi từ 0.778 lên **2.106**, trong khi teacher là 2.221. Nó hội tụ về
lỗi của teacher triệt để hơn là về thành công. Nếu worst-patient FAR là ràng
buộc lâm sàng cứng thì chọn L4 thay vì L8.

### Thí nghiệm chưa ai chạy, và nó là bước tiếp theo rõ ràng

**L4 + L8 chưa từng được kết hợp.** L4 cho FAR tốt nhất (0.1904), L8 cho
sensitivity tốt nhất (0.9489). Hai đòn bẩy độc lập — L4 đổi tiêu chí chọn
checkpoint, L8 đổi hàm mất mát — nên về nguyên tắc cộng dồn được. Đây là cơ hội
duy nhất còn lại có thể đưa cả M2 và M3 qua cùng lúc, và nó chỉ tốn 198 fold
(~3.3 h nếu chia ba máy).


---

## 6. Rủi ro phải nêu: INT8 chưa từng được đo

`A4` (mất mát do lượng tử hoá ≤ 0.5pp) **chưa có số nào**. Ở mức 0.9489 so với
mục tiêu 0.95, một mất mát 0.5pp đưa kết quả xuống 0.944.

Nhóm phần cứng nên biết rằng họ đang xây cho một model **chưa biết hành vi INT8**.
Điều này không chặn việc thiết kế datapath và bộ nhớ — các con số ở §2–§3 đều
đã tính theo INT8 — nhưng nó chặn việc công bố con số độ chính xác cuối cùng.

## 7. Việc bắt đầu được ngay, không cần chờ

- Sizing PE array và phân bổ SRAM cho **khoảng 8.6–18.2 KiB** (cả ba ứng viên nằm trong khoảng này).
- Kiến trúc streaming line buffer: quy tắc `(k−1)·dilation+1` giống nhau cho mọi ứng viên; chỉ số kênh đổi.
- Giao diện: 1 kênh, 256 Hz, cửa sổ 4 s, quyết định mỗi 1 s → **duty cycle rất thấp**, nên năng lượng mỗi giờ bị chi phối bởi công suất tĩnh chứ không phải động.
- **Đặt mua shunt đo công suất ngoài.** PYNQ-Z2 không có cảm biến on-board, và đây là hạng mục có thời gian chờ vật lý — nếu để tới lúc cần thì nó thành đường găng.
