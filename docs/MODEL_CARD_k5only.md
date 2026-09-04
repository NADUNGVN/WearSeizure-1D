# Model chốt cho đề tài sinh viên

**`wearseizure1d_k5only`**, trọng số từ recipe **L1 + L8** (row 43 của
`EXPERIMENT_LOG_G1a.md`). Đây là cấu hình có **sensitivity cao nhất** trong mọi
thứ dự án đã đo mà vẫn nằm trong ngân sách MAC.

> **Tạm chốt.** Kiến trúc đã cố định và sẽ không đổi — thang dung lượng đã chạy
> xong, không phương án nào tốt hơn. Trọng số còn có thể đổi (xem §6), nhưng
> **đổi trọng số không đụng một dòng RTL nào**: cùng layer, cùng MACs, cùng line
> buffer.

---

## 1. Bài toán

| | |
|---|---|
| Đầu vào | **1 kênh EEG**, 256 Hz |
| Cửa sổ | **4 giây = 1024 mẫu** |
| Bước trượt | 1 giây → một quyết định mỗi giây |
| Đầu ra | 2 lớp (ictal / non-ictal) |
| Tiền xử lý | bandpass **nhân quả** 1–30 Hz, chuẩn hoá affine |

Tiền xử lý là **nhân quả** (một chiều, `lfilter`), không phải `filtfilt` — bắt
buộc, vì thiết bị đeo không thể nhìn về tương lai.

---

## 2. Cấu trúc tổng thể

```
    đầu vào   1 × 1024                    (4 s @ 256 Hz, một kênh)
       │
   ┌───▼────┐  Stem: Conv k7, stride 2 + BN + ReLU
   │  stem  │
   └───┬────┘  8 × 512
       │
   ┌───▼────┐  B1: depthwise-separable k5, stride 2
   │   b1   │
   └───┬────┘  16 × 256
       │
   ┌───▼────┐  B2: depthwise k5 dilation 1, stride 2 → pointwise
   │   b2   │
   └───┬────┘  24 × 128
       │
   ┌───▼────┐  B3: depthwise k5 dilation 2, stride 2 → pointwise
   │   b3   │
   └───┬────┘  32 × 64
       │
   ┌───▼────┐  B4: depthwise k5 dilation 4, stride 2 → pointwise
   │   b4   │
   └───┬────┘  48 × 32
       │
   ┌───▼────┐  Context: 2 × depthwise-separable k5,
   │context │             dilation 8 rồi 16, KHÔNG stride
   └───┬────┘  64 × 32
       │
   ┌───▼────┐  Global Average Pooling trên trục thời gian
   │  GAP   │
   └───┬────┘  64
       │
   ┌───▼────┐  Linear(64, 2)
   │   FC   │
   └───┬────┘  2
       ▼
     logits
```

**Ba ý tưởng thiết kế:**

1. **Depthwise-separable ở mọi nơi.** Mỗi block tách thành conv theo kênh
   (`groups = in_channels`) rồi conv 1×1 trộn kênh. Đây là lý do model chỉ có
   11 786 tham số.
2. **Dilation tăng dần 1 → 2 → 4 → 8 → 16.** Chuỗi bị giảm mẫu 2× mỗi block,
   nên dilation tăng để trường tiếp nhận phủ hết 4 giây mà không cần kernel lớn.
3. **GAP thay vì flatten.** Đầu ra không phụ thuộc độ dài chuỗi và không có lớp
   fully-connected lớn — FC cuối chỉ 64×2 = 128 tham số.

Biến thể này là **`k5_only`**: `MultiScaleDilatedBlock` vốn có hai nhánh song
song (k3 và k5) nhưng ablation đã cho thấy nhánh k3 không đóng góp gì, nên nó bị
tắt. RTL vẫn nên dựng bộ sinh tap dùng chung cho k3/k5 nếu muốn khả lập trình.

---

## 3. Bảng từng layer

Đầu vào 1 × 1024. Sinh bằng `python scripts/hardware_spec.py wearseizure1d_k5only`
— không con số nào chép tay.

| layer | kiểu | in | out | k | s | dil | out_len | taps | buffer | MACs | trọng số |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `stem.0` | conv | 1 | 8 | 7 | 2 | 1 | 512 | 7 | 7 | 28 672 | 56 |
| `b1.depthwise` | depthwise | 8 | 8 | 5 | 2 | 1 | 256 | 5 | 40 | 10 240 | 40 |
| `b1.pointwise` | conv 1×1 | 8 | 16 | 1 | 1 | 1 | 256 | 1 | 8 | 32 768 | 128 |
| `b2.branch_k5` | depthwise | 16 | 16 | 5 | 2 | 1 | 128 | 5 | 80 | 10 240 | 80 |
| `b2.pointwise` | conv 1×1 | 16 | 24 | 1 | 1 | 1 | 128 | 1 | 16 | 49 152 | 384 |
| `b3.branch_k5` | depthwise | 24 | 24 | 5 | 2 | **2** | 64 | 9 | 216 | 7 680 | 120 |
| `b3.pointwise` | conv 1×1 | 24 | 32 | 1 | 1 | 1 | 64 | 1 | 24 | 49 152 | 768 |
| `b4.branch_k5` | depthwise | 32 | 32 | 5 | 2 | **4** | 32 | 17 | 544 | 5 120 | 160 |
| `b4.pointwise` | conv 1×1 | 32 | 48 | 1 | 1 | 1 | 32 | 1 | 32 | 49 152 | 1 536 |
| `context.0.depthwise` | depthwise | 48 | 48 | 5 | 1 | **8** | 32 | 33 | 1 584 | 7 680 | 240 |
| `context.0.pointwise` | conv 1×1 | 48 | 64 | 1 | 1 | 1 | 32 | 1 | 48 | 98 304 | 3 072 |
| `context.1.depthwise` | depthwise | 64 | 64 | 5 | 1 | **16** | 32 | **65** | **4 160** | 10 240 | 320 |
| `context.1.pointwise` | conv 1×1 | 64 | 64 | 1 | 1 | 1 | 32 | 1 | 64 | 131 072 | 4 096 |

Sau đó: **GAP** (32 → 1) → **Linear(64, 2)**.

Mỗi conv đi kèm **BatchNorm + ReLU**. BatchNorm **gấp vào conv khi suy luận**
thành scale/bias mỗi kênh — datapath **không cần khối BN riêng**.

---

## 4. Tham số cho thiết kế phần cứng

Toàn bộ mục này sinh bằng `python scripts/hardware_spec.py wearseizure1d_k5only`,
đọc trực tiếp từ model bằng forward hook. Không con số nào chép tay.

### 4.1 Trọng số

| | |
|---|---:|
| tham số | **11 786** |
| bộ nhớ trọng số INT8 | **11 786 B = 11.5 KiB** |
| bias | không có — mọi conv đặt `bias=False` |
| BatchNorm | **gấp vào conv khi suy luận** → scale + bias mỗi kênh, không cần khối riêng |

Trọng số nằm **hoàn toàn on-chip** (mục tiêu H2), không cần truy cập DRAM.

### 4.2 Bộ nhớ kích hoạt — hai kiểu kiến trúc, hai con số

Đây là chỗ dễ báo cáo thiếu nhất. **Không có một con số "activation memory" duy
nhất**; nó phụ thuộc kiểu accelerator:

| kiểu | cần giữ gì | bộ nhớ |
|---|---|---:|
| **Fully streaming** (pipeline, mọi layer chạy đồng thời trên dòng mẫu) | mọi line buffer sống cùng lúc, **không bao giờ vật chất hoá feature map** | **6 823 B = 6.7 KiB** |
| **Layer sequential** (một layer một lượt) | một line buffer + **cả feature map vào và ra** | **8 256 B = 8.1 KiB** |

Con số layer-sequential là **cực đại lấy theo từng layer**, không phải tổng của
các trường hợp xấu nhất. Cặp feature map rộng nhất là **6 144 B** (ở
`b1.depthwise`, buffer riêng chỉ 40 B) còn line buffer lớn nhất là **4 160 B**
(ở `context.1.depthwise`, feature map nhỏ). Hai cái ở **hai layer khác nhau**,
nên cộng chúng lại sẽ thừa 2 048 B. Cực đại thật là **8 256 B** tại
`context.1.depthwise`: 2 048 vào + 2 048 ra + 4 160 buffer.

**Mạng không có skip connection.** `WearSeizure1D.forward` và
`DepthwiseSeparableConv1d.forward` thuần tuần tự, nên không có tensor nào phải
sống qua ranh giới block. (`MultiScaleDilatedBlock` có `torch.cat` hai nhánh —
nhưng chỉ ở chế độ `multi_scale`; biến thể `k5_only` đã chốt chỉ một nhánh. Thiết
kế khả lập trình nếu phải chạy cả bản multi-scale thì cần thêm tối đa 4 096 B cho
hai nhánh cùng sống.)

Tách depthwise và pointwise thành hai lượt **làm giảm** đỉnh chứ không tăng: ở
mức block, `stem → b1` cần 4 096 + 4 096 = 8 192 B, còn tách ra thì giải phóng
được output của stem trước khi cấp phát output của b1, còn 6 144 B.

**Tổng SRAM on-chip:**

| kiểu | trọng số | kích hoạt | **tổng** |
|---|---:|---:|---:|
| fully streaming | 11.5 KiB | 6.7 KiB | **18.2 KiB** |
| layer sequential | 11.5 KiB | 8.1 KiB | **19.6 KiB** |

Cả hai đều **dưới 1 %** BRAM của XC7Z020 (140 × 36 Kb ≈ 630 KiB), nên mục tiêu
H4 (< 10 % tài nguyên) không bị bộ nhớ chặn.

### 4.3 Line buffer từng layer

Công thức: `(kernel_size − 1) × dilation + 1` mẫu, nhân `in_channels`.
**Dilation, không phải kernel size, quyết định chi phí.**

| layer | k | dilation | taps | in_ch | buffer (B) | padding mỗi bên |
|---|--:|--:|--:|--:|--:|--:|
| `stem.0` | 7 | 1 | 7 | 1 | 7 | 3 |
| `b1.depthwise` | 5 | 1 | 5 | 8 | 40 | 2 |
| `b2.branch_k5` | 5 | 1 | 5 | 16 | 80 | 2 |
| `b3.branch_k5` | 5 | 2 | 9 | 24 | 216 | 4 |
| `b4.branch_k5` | 5 | 4 | 17 | 32 | 544 | 8 |
| `context.0.depthwise` | 5 | 8 | 33 | 48 | 1 584 | 16 |
| **`context.1.depthwise`** | 5 | **16** | **65** | 64 | **4 160** | **32** |
| các conv 1×1 | 1 | 1 | 1 | — | 8–64 | 0 |

`context.1.depthwise` một mình chiếm **61 %** tổng line buffer. Nó là k5 dilation
16 chạy trên chuỗi dài **32 mẫu** với padding **32 mỗi bên** — tức 64 mẫu padding
bao quanh 32 mẫu thật, phần lớn tap đọc số 0.

> **Đã thử cắt và không miễn phí.** Biến thể `ctx16` (context 64→16) tiết kiệm
> 37 % MACs nhưng **mất 2.33 pp sensitivity macro**. Xem `EXPERIMENT_LOG_G1a.md`
> §2h. Đừng "tối ưu" lại chỗ này mà không đọc.

### 4.4 Feature map sau mỗi layer

| sau layer | hình dạng | INT8 bytes |
|---|---|--:|
| đầu vào | 1 × 1024 | 1 024 |
| `stem` | 8 × 512 | 4 096 |
| `b1` | 16 × 256 | 4 096 |
| `b2` | 24 × 128 | 3 072 |
| `b3` | 32 × 64 | 2 048 |
| `b4` | 48 × 32 | 1 536 |
| `context.0` | 64 × 32 | 2 048 |
| `context.1` | 64 × 32 | 2 048 |
| GAP | 64 | 64 |

Cặp liền kề lớn nhất là `stem → b1`: 4 096 + 4 096 = **8 192 B** ở mức layer, và
6 144 B ở mức conv riêng lẻ (con số dùng ở §4.2, vì mỗi conv trong một block là
một bước tính riêng).

### 4.5 Tính toán và định cỡ PE array

| | |
|---|---:|
| MACs `thop` (số dùng trong bài báo) | 585 920 |
| **MACs conv+fc — số accelerator thật sự phải phát** | **489 600** |
| tần suất suy luận | **1 lần / giây** (cửa sổ 4 s, trượt 1 s) |
| ngân sách độ trễ (H3) | ≤ **2 ms** |

Hai quy ước MAC lệch nhau vì `thop` tính cả BatchNorm và phép elementwise; BN
gấp vào conv khi suy luận nên accelerator không phát chúng. **Nêu cả hai trong
luận văn** để không ai đọc nhầm.

Suy ra yêu cầu thông lượng: 489 600 MAC / 2 ms = **245 MMAC/s**.

| clock | MAC/chu kỳ cần | thời gian mỗi suy luận | duty cycle |
|---:|---:|---:|---:|
| 100 MHz | **2.45** → chọn **4** | 1.22 ms | **0.12 %** |
| 100 MHz | 8 | 0.61 ms | 0.06 % |
| 50 MHz | **4.9** → chọn **8** | 1.22 ms | 0.12 % |

Một PE array **4–8 MAC** là đủ. Đây không phải bài toán bị chặn bởi compute.

### 4.6 Hệ quả năng lượng — dễ kết luận ngược

Duty cycle **~0.1 %** nghĩa là accelerator ngủ hơn 99.9 % thời gian. **Năng lượng
mỗi giờ bị chi phối bởi công suất tĩnh, không phải động.**

Hệ quả: **tối ưu MACs không phải đòn bẩy năng lượng chính.** Clock gating,
power gating và dòng rò khi nhàn rỗi mới là thứ quyết định thời lượng pin. Một
luận văn accelerator rất dễ dành toàn bộ công sức vào việc giảm MACs và không
thu được gì về năng lượng.

### 4.7 Giao diện

| | |
|---|---|
| đầu vào | 1 kênh, 256 Hz, một mẫu mỗi 3.906 ms |
| cửa sổ | 1024 mẫu (4 s) |
| bước trượt | 256 mẫu (1 s) → một quyết định mỗi giây |
| tiền xử lý | bandpass **nhân quả** 1–30 Hz + chuẩn hoá affine |
| đầu ra | 2 logits → hậu xử lý EMA + ngưỡng trễ |

Tiền xử lý phải **nhân quả** (`lfilter` một chiều, reset trạng thái mỗi bản ghi),
không phải `filtfilt`. Thiết bị đeo không nhìn được về tương lai. Nếu bộ lọc được
làm cứng thì nó cũng nằm trong ngân sách tài nguyên và phải tính vào.

### 4.8 ĐÃ ĐO: định dạng số đã chốt là **DFP16**

Đã quét 5 định dạng trên 66 fold × 3 seed (`EXPERIMENT_LOG_G1a.md` §2m).

**Chốt: `dfp16` — dynamic fixed point 16 bit, scale per-channel.**

| | mất so với FP32 | CI 95% | mất tối đa dữ liệu cho phép |
|---|---:|---|---:|
| **`dfp16`** | **−0.10 pp** | **[−0.25, 0.00]** | **0.25 pp — trong cổng 0.5** |
| `dfp8` | −0.51 pp | [−1.92, 0.00] | 1.92 pp — vượt cổng |
| `int8` | −0.64 pp | [−2.06, +1.31] | 2.06 pp — vượt cổng |
| `int16` | −0.89 pp | [−1.90, 0.00] | 1.90 pp — vượt cổng |

**Điểm ước lượng của cả bốn đều nằm trong nhiễu** — `int16` còn "tệ hơn" `int8`,
điều không thể là hiệu ứng thật. Cái quyết định là **độ rộng khoảng**: `dfp16` là
định dạng duy nhất mà **trường hợp xấu nhất dữ liệu còn cho phép** vẫn nằm trong
cổng 0.5 pp.

Giá phải trả cho khoảng chặt hơn: **18 KiB SRAM** (18.2 → 36.3 KiB, tức 3.2% →
6.5% BRAM) và **không thêm DSP nào** — DSP48E1 là bộ nhân 25×18 nên phép 16×16
vừa đúng một DSP như phép 8×8.

Bộ nhớ ở §4.2 vì thế phải đọc ở cột **INT16/DFP16**: **36.3 KiB** streaming,
39.1 KiB layer-sequential.

---

## 5. Chất lượng

3 seed × 66 fold, giao thức `patient_specific_loso_edf`, **185.0 h** exposure,
13 bệnh nhân, 77 cơn.

| chỉ số | giá trị |
|---|---:|
| **sensitivity mức sự kiện (macro)** | **0.9489** |
| **sensitivity mức sự kiện (micro)** | **0.9567** |
| FAR/h | 0.2937 |
| độ trễ phát hiện trung bình | 17.75 s |
| worst-patient sensitivity (≥5 cơn) | 0.9333 |
| worst-patient FAR/h | 2.1065 |

Con số mức cửa sổ để so với văn liệu (từ A7 bậc C, cùng giao thức sạch, model
train từ đầu): sensitivity 0.6033, specificity 0.9943, **accuracy 0.9888**,
**ictal prevalence 1.42 %**.

> **Accuracy không bao giờ được đứng một mình.** Ở prevalence này, một model
> luôn trả lời "không có cơn" đạt **98.58 %**. Prevalence phải nằm ngay cột bên
> cạnh.

---

## 6. Điều còn có thể đổi — và điều không

**Kiến trúc: cố định.** Thang dung lượng đã chạy xong; `ctx16` kém hơn 2.33pp,
`wide` ngang bằng (micro trùng tới 16 chữ số thập phân). Không còn gì buộc thiết
kế lại datapath.

**Trọng số: có thể còn đổi.** Ba recipe cùng kiến trúc:

| recipe | sens macro | FAR/h | worst-pt FAR |
|---|---:|---:|---:|
| L1 | 0.9358 | 0.2216 | 0.7785 |
| L1 + L4 | 0.9380 | **0.1904** | **0.5701** |
| **L1 + L8 (đang chốt)** | **0.9489** | 0.2937 | 2.1065 |

L8 tốt nhất về sensitivity nhưng tệ nhất về worst-patient FAR — chưng cất đã sao
chép luôn chế độ hỏng của teacher trên bệnh nhân chb23. Nếu worst-patient FAR
trở thành ràng buộc thì đổi sang L4.

**Đổi trọng số không tốn một dòng RTL.** Cùng layer, cùng MACs, cùng line
buffer — chỉ khác giá trị trong bộ nhớ trọng số.

---
