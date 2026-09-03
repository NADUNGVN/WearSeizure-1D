# Benchmark cho luận văn sinh viên

Đề tài: *Design of Programmable Hardware for 1-D CNN on SoC in EEG-based Epileptic
Seizure Detection*.

Tài liệu này nói **luận văn cần báo cáo những con số nào** và **lấy chúng ở đâu**. Nó khác
`RESEARCH_REALITY_CHECK.md` §12.6 ở một điểm quan trọng: §12.6 viết cho một accelerator **cố
định** cho một mạng duy nhất, còn đề tài này là **khả lập trình**. Khối 3 dưới đây là phần §12.6
không có, và cũng chính là phần đóng góp riêng của luận văn.

---

## Khối 1 — Chất lượng model: bối cảnh, không phải đóng góp

Luận văn là về phần cứng. Khối này tồn tại để chứng minh thứ được tăng tốc là đáng tin, và để trả
lời câu hỏi chắc chắn sẽ bị hỏi lúc bảo vệ.

| chỉ số | giá trị | lấy ở đâu |
|---|---|---|
| accuracy mức cửa sổ | *(chờ A7 bậc C)* | `summarise_leaky_repro.py` |
| sensitivity / specificity mức cửa sổ | *(chờ A7 bậc C)* | như trên |
| **prevalence ictal** | **~0.5 %** | phải in **cạnh** accuracy, luôn luôn |
| sensitivity mức sự kiện | **0.9489** (macro, 3 seed) | `EXPERIMENT_LOG_G1a.md` row 43 |
| FAR/h | 0.2937 | row 43 |
| độ trễ phát hiện trung bình | 17.75 s | row 43 |
| exposure đánh giá | **185.0 h** | mọi row |

**Quy tắc không được phá:** accuracy **không bao giờ** đứng một mình. Ở tỉ lệ 0.5 % ictal, một model
luôn trả lời "không có cơn" đạt **99.5 % accuracy, 0 % sensitivity**. Prevalence phải nằm ngay cột
bên cạnh. Đây là chính sách đã ghi trong `src/wearseizure/eval/metrics_segment.py` từ đầu.

### Câu hỏi sẽ bị hỏi, và cách trả lời bằng dữ liệu

> *"Sao 93 % của em thấp hơn 99 % trong các bài báo?"*

Trả lời bằng thang A7, không bằng lý thuyết:

| bậc | giao thức | sensitivity |
|---|---|---|
| A | chia ngẫu nhiên theo cửa sổ, như văn liệu | *(chờ)* — dự kiến ~0.99 |
| C | chia theo bản ghi, không rò rỉ | *(chờ)* |
| D | như C, nhưng đếm theo **sự kiện** | 0.8756 (k5only, train từ đầu) |

Kèm con số quyết định: dưới cách chia của văn liệu, **99.6 % cửa sổ test chia sẻ mẫu với một cửa sổ
train** (đo trên chb01, dữ liệu thật). Cửa sổ 4 s trượt 1 s thì hai cửa sổ liền nhau trùng 75 % mẫu.

Đây biến một điểm yếu biểu kiến thành một đóng góp về phương pháp.

---

## Khối 2 — Phần cứng: đóng góp chính

Lấy nguyên `RESEARCH_REALITY_CHECK.md` §12.6.

| # | mục tiêu | ngưỡng |
|---|---|---|
| H1 | RTL khớp bit-exact với INT8 reference | **0 sai khác / 10 000 cửa sổ** |
| H2 | Trọng số hoàn toàn on-chip | 0 truy cập DRAM cho trọng số |
| H3 | Độ trễ suy luận | ≤ 2 ms |
| H4 | Tài nguyên | < 10 % LUT / DSP / BRAM của XC7Z020 |
| H5 | Công suất | **đo thật** bằng shunt ngoài, tách tĩnh/động |
| H6 | Hiệu quả | pJ/MAC, và so với cùng model chạy trên ARM Cortex-A9 của chính board |
| H7 | Điểm so sánh | ≥ 2: (a) ARM cùng board, (b) ≥ 1 accelerator đã công bố |

Dấu chân đã đo, sinh từ chính model (`scripts/hardware_spec.py`):

| | `k5only` | `ctx16` | `wide` |
|---|---:|---:|---:|
| MACs triển khai (conv+fc) | 489 600 | 285 216 | 513 568 |
| trọng số INT8 | 11.5 KiB | 5.0 KiB | 9.2 KiB |
| line buffer INT8 | 6.7 KiB | 3.6 KiB | 4.8 KiB |
| **tổng SRAM** | **18.2 KiB** | **8.6 KiB** | **14.0 KiB** |

Hai lưu ý phải viết vào luận văn:

- Bài báo dùng **585 920 MACs** (`thop`, có tính BatchNorm) nhưng accelerator chỉ phát **489 600**,
  vì **BatchNorm gấp vào conv khi suy luận**. Nêu rõ cả hai để không ai đọc nhầm.
- **Duty cycle rất thấp** — cửa sổ 4 s, quyết định mỗi 1 s. Năng lượng mỗi giờ do **công suất tĩnh**
  chi phối, không phải động. Tối ưu MACs vì thế **không** phải đòn bẩy năng lượng chính, và một luận
  văn accelerator dễ mắc lỗi ngược lại.

---

## Khối 3 — Tính khả lập trình: phần §12.6 không có

Đây là điểm phân biệt đề tài này với "một accelerator cho một mạng". Không có khối này, luận văn chỉ
chứng minh được nửa cái tên của nó.

| # | mục tiêu | vì sao |
|---|---|---|
| P1 | **Envelope tham số được phủ**, khai báo tường minh | Nói "khả lập trình" mà không nêu miền hợp lệ thì không kiểm chứng được |
| P2 | **Chạy đúng ≥ 3 hình dạng mạng khác nhau** trên cùng bitstream, mỗi cái bit-exact | Đây là bằng chứng của "programmable"; một mạng thì chỉ là hardwired |
| P3 | **Chi phí tái cấu hình**: thời gian nạp cấu hình + trọng số, không nạp lại bitstream | Nếu phải nạp lại bitstream thì không phải khả lập trình mà là tổng hợp lại |
| P4 | **Hiệu suất sử dụng PE theo từng hình dạng** | Một thiết kế linh hoạt thường lãng phí PE ở mạng nhỏ — đo và báo cáo, đừng giấu |
| P5 | So sánh với **chính nó ở cấu hình cố định** | Trả lời "linh hoạt tốn bao nhiêu" — câu phản biện chắc chắn có |

### Envelope đề xuất (P1)

Đo trên **cả năm** kiến trúc dự án từng có, bằng `scripts/hardware_spec.py`:

| tham số | tối đa thực tế | thiết kế cho | biên |
|---|---:|---:|---:|
| số layer (conv + FC) | 17 | **32** | 1.9× |
| kênh mỗi layer | 72 | **128** | 1.8× |
| kernel size | 7 | **≤ 7** | — |
| dilation | 16 | **≤ 16** | — |
| tap trong một cửa sổ | 65 | **128** | 2.0× |
| line buffer tổng | 6.9 KiB | **16 KiB** | 2.3× |
| bộ nhớ trọng số INT8 | 14.5 KiB | **32 KiB** | 2.2× |
| MACs mỗi cửa sổ | 650 880 | **1 M** | 1.5× |
| độ dài chuỗi | 1024 | **≤ 2048** | 2.0× |

Tổng SRAM 48 KiB — rất nhỏ trên Zynq-7020. **Phần khó của đề tài không phải nhét vừa**, mà là cơ
chế khả lập trình: instruction/config stream, datapath tham số hoá theo kênh và dilation, cấp phát
line buffer động.

### Bộ mạng để chứng minh P2

Dự án đã có sẵn năm cấu hình đo được, dùng luôn làm tập chứng minh — không phải bịa ra mạng giả:

| mạng | layer | kênh tối đa | dilation | MACs |
|---|--:|--:|---|--:|
| `wearseizure1d_k5only` | 14 | 64 | 1,2,4,8,16 | 489 600 |
| `wearseizure1d_k5only_ctx16` | 14 | 48 | 1,2,4,8,16 | 285 216 |
| `wearseizure1d_k5only_wide` | 14 | 72 | 1,2,4,8,16 | 513 568 |
| `wearseizure1d` (mặc định) | 17 | 64 | 1,2,4,8,16 | 650 880 |
| `wearseizure1d_k3only` | 14 | 64 | 1,8,16 | 480 384 |

Chúng khác nhau ở **số layer** (14 vs 17), **số kênh** (48–72), **kernel** (k3/k5/k7) và **tập
dilation** — tức phủ đúng bốn chiều mà P1 khai báo. Ba trong số này đã có kết quả huấn luyện thật,
nên P2 kiểm chứng được bằng dữ liệu thật chứ không bằng vector ngẫu nhiên.

### Ba con số đáng nói với sinh viên trước khi bắt đầu

1. **Dilation, không phải kernel size, quyết định chi phí buffer.** Quy tắc `(k−1)·dilation+1`.
   Cùng k5: dilation 1 cần 5 tap, dilation 16 cần **65**.
2. **`context.1.depthwise` chiếm 61 % toàn bộ line buffer** của `k5only`, vì k5 dilation 16 trải 65
   mẫu trên chuỗi dài **32** — phần lớn tap đọc padding. Layer kém hiệu quả nhất thiết kế.
3. **BatchNorm không cần khối riêng** — gấp vào conv thành scale/bias mỗi kênh.

---

## Việc chặn

**Mất mát INT8 chưa từng được đo (A4).** Ở mức 0.9489 so với mục tiêu 0.95, một mất mát 0.5 pp đưa
xuống 0.944. Điều này **không chặn** thiết kế datapath và bộ nhớ — mọi con số ở Khối 2 và 3 đều đã
tính theo INT8 — nhưng nó **chặn con số độ chính xác cuối cùng** mà luận văn công bố.

**Shunt đo công suất có thời gian chờ vật lý.** PYNQ-Z2 không có cảm biến on-board, nên H5 không đo
được nếu không có nó. Đây là việc mua sắm, đặt được ngay, và nếu để muộn sẽ thành đường găng.
