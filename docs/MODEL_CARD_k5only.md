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

## 4. Tổng kết tài nguyên

| | |
|---|---:|
| Tham số = bộ nhớ trọng số INT8 | **11 786 B ≈ 11.5 KiB** |
| MACs `thop` (số dùng cho bài báo) | 585 920 |
| **MACs conv+fc (số accelerator phải phát)** | **489 600** |
| Line buffer INT8 | **6 823 B ≈ 6.7 KiB** |
| **Tổng SRAM on-chip** | **≈ 18.2 KiB** |

Hai quy ước MAC lệch nhau vì `thop` tính cả BatchNorm và phép elementwise. Nêu
cả hai trong luận văn để không ai đọc nhầm.

### Ba điều về bộ nhớ mà thiết kế phần cứng phải biết

1. **Line buffer = `(k−1)·dilation + 1` mẫu × in_channels.** **Dilation**, không
   phải kernel size, quyết định chi phí. Cùng k5: dilation 1 cần 5 tap, dilation
   16 cần **65**.
2. **`context.1.depthwise` chiếm 61 % toàn bộ line buffer** (4 160 / 6 823). Nó
   là k5 dilation 16 trải **65 mẫu trên chuỗi dài 32** — phần lớn tap đọc
   padding. Layer kém hiệu quả nhất thiết kế.
3. Đã thử cắt nó (`ctx16`) và **mất 2.33pp sensitivity**. Không phải tiết kiệm
   miễn phí. Xem `EXPERIMENT_LOG_G1a.md` §2h.

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

## 7. Việc chưa đo, phải nêu trong luận văn

**Mất mát do lượng tử hoá INT8 chưa từng được đo.** Mọi con số tài nguyên ở §4
đã tính theo INT8, nên thiết kế datapath và bộ nhớ tiến hành được. Nhưng con số
độ chính xác cuối cùng mà luận văn công bố **phải chờ** phép đo đó.
