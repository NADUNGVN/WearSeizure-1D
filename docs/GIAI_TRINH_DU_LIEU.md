# Giải trình về dữ liệu: vì sao một kênh, vì sao 13 ca

Tài liệu này trả lời ba câu hỏi về dữ liệu của đề tài, kèm dẫn chứng từ y văn và
từ các thí nghiệm đã chạy:

1. Vì sao huấn luyện trên **một kênh** thay vì 23 kênh của CHB-MIT?
2. Vì sao đánh giá trên **13 ca** thay vì cả 24 ca?
3. Trong **44 GB** dữ liệu gốc, thực sự dùng bao nhiêu?

---

## 1. Một kênh — và đây là lựa chọn có căn cứ, không phải cắt bớt

### 1.1 Bài tham chiếu đã tự kiểm chứng điều này

Chung et al. 2024 (Frontiers in Neurology, doi:10.3389/fneur.2024.1389731) —
chính là bài mà đề tài dùng làm baseline — đã dựng **ba bộ phát hiện trên cùng
một dữ liệu** và so với nhau:

| Cấu hình | Độ nhạy | FAR/h |
|---|--:|--:|
| 18 kênh (montage đầy đủ) | 100 % | 0,30 ± 0,47 |
| 4 kênh (Fp1-F3, Fp2-F4, P7-O1, P8-O2) | 97,05 ± 9,23 % | 0,40 ± 0,77 |
| **1 kênh** (chọn theo xác nhận của bác sĩ) | **99,62 ± 1,39 %** | **0,22 ± 0,34** |

**Bản một kênh có tỷ lệ báo động giả thấp nhất trong cả ba, và độ nhạy cao hơn
bản bốn kênh.** Thêm kênh không mua được gì trên bài toán này.

Lý do hợp lý về mặt sinh lý: cơn động kinh khởi phát **khu trú**. Khi kênh được
chọn đúng vị trí khởi phát — Chung et al. chọn thủ công theo xác nhận của bác sĩ
thần kinh, không dùng thuật toán chọn kênh — thì 17 kênh còn lại chủ yếu đóng
góp nhiễu chứ không đóng góp tín hiệu.

### 1.2 Ba thí nghiệm của đề tài đều nói cùng một điều

| Thí nghiệm | Thêm gì vào lúc huấn luyện | Kết quả |
|---|---|---|
| **L1** | thêm **bệnh nhân**, vẫn 1 kênh mỗi người | bước nhảy lớn nhất của dự án |
| **L5** | thêm **giờ ghi**: 553 h → 2085 h (×3,7) | không cải thiện; biến thể 1 vị trí **−4,34 pp**, CI 95 % [−8,61; −0,86] |
| **L3** | teacher đọc **18–26 kênh**, chưng cất sang student 1 kênh | **âm** |
| **L8** | teacher **cùng 1 kênh**, chỉ lớn hơn về dung lượng | +1,31 pp, CI [−2,52; +6,03] (chưa đủ ý nghĩa) |

Chi tiết L5 (66 fold × 3 seed, mỗi cấu hình so với đối chứng của **chính nó** —
cùng kiến trúc, cùng seed, cùng hậu xử lý, chỉ khác corpus):

| Corpus tiền huấn luyện | Độ nhạy | FAR/h |
|---|--:|--:|
| 13 ca (đối chứng) | **0,9358 ± 0,0304** | **0,2261** |
| + 11 ca, 4 vị trí | 0,9348 ± 0,0153 | 0,2318 |
| + 11 ca, 1 vị trí | 0,8924 ± 0,0116 | 0,2547 |

Đối chứng thắng cả sáu ô (hai kiến trúc × ba chỉ số).

**Cơ chế giải thích cả bốn dòng:** một soft target chỉ bắt chước được nếu student
**về nguyên tắc tính ra được nó**. Teacher đa kênh đưa ra độ tin cậy dựa trên
những kênh mà student sẽ không bao giờ nhìn thấy, nên student bị kéo về một con
số nó không có đường nào suy ra. Nói gọn: *chưng cất giúp khi ưu thế của teacher
là **dung lượng**, và có hại khi ưu thế là **thông tin***.

### 1.3 Ràng buộc phần cứng, đo được

Thiết bị đeo có **một cặp điện cực và một ADC**. Ngoài ra, với vi kiến trúc hiện
tại (`CNN_1D_Core.v`: `FM_BANK_NUM = 16`, `FM_AWIDTH = 10`):

- Bộ nhớ feature map mỗi buffer ping/pong = **16 × 1024 = 16.384 ô**.
- Cửa sổ 23 kênh × 1024 mẫu cần **23.552 ô** → **không nằm vừa**, thiếu 7.168 ô.
- Khối lượng tính tăng từ **585.920** lên **1.216.704 MACs**, vượt mục tiêu
  "dưới 1 triệu MACs" của đề cương.

*Ghi chú kỹ thuật:* bản thân bộ tăng tốc **có thể** tính đa kênh — trường `IN_CH`
rộng 10 bit, tới 1023 kênh. Chỗ chặn là bộ nhớ trên chip và ngân sách tính toán,
và trên hết là cảm biến không tồn tại trên thiết bị đeo.

---

## 2. Mười ba ca — căn cứ, và điểm yếu cần thừa nhận

### 2.1 Không phải do thiếu kênh

**Mọi montage chuẩn của CHB-MIT đều mang đủ cả bốn vị trí đeo được**
(`Fp1-F3`, `P7-O1`, `P8-O2`, `Fp2-F4`). Cả 24 ca đều có sẵn các kênh này. Giới
hạn 13 ca đến từ chỗ khác: đó là những ca mà Chung et al. 2024 **đã xác nhận lâm
sàng** rằng khởi phát cơn quan sát được từ **một** vị trí đeo cụ thể.

Phân bổ hiện tại:

| Kênh | Bệnh nhân |
|---|---|
| `P7-O1` | chb02, chb05, chb10, chb11, chb15 |
| `Fp1-F3` | chb03, chb07, chb08, chb22, chb23 |
| `P8-O2` | chb01, chb04, chb17 |

Xác nhận vị trí là yêu cầu để **chấm điểm** một bộ phát hiện một kênh — nếu cơn
không nhìn thấy được ở kênh đó thì phép đo không nói lên điều gì về model. Nhưng
nó **không** phải yêu cầu để **dạy** model biết sóng ictal trông như thế nào; đó
chính là lý do lever L5 được thiết kế để mở rộng corpus tiền huấn luyện ra ngoài
13 ca trong khi giữ nguyên tập đánh giá.

### 2.2 Điểm yếu phải thừa nhận

13 ca **là một tập con thuận lợi**. Lý do chọn có căn cứ trích dẫn được, nhưng
căn cứ đó không loại trừ được nghi ngờ chọn lọc, và phản biện sẽ hỏi đúng câu
này.

**Hướng xử lý đề xuất:** chạy thêm giao thức trên **đủ 24 ca**, vẫn một kênh.
Vấn đề duy nhất phải giải là 11 ca ngoài Appendix A chưa biết dùng vị trí nào
trong bốn — cách sạch là **chọn vị trí trên tập train/val của chính bệnh nhân
đó, không bao giờ chạm tập test**, đúng như một giai đoạn hiệu chỉnh khi đeo máy
thật.

Dự kiến độ nhạy trên 24 ca sẽ **thấp hơn** 13 ca, vì 11 ca kia không có bằng
chứng lâm sàng rằng cơn nhìn thấy được từ vị trí đeo. Đó vẫn là kết quả có giá
trị và nên báo cáo song song hai bảng: khoảng cách giữa hai con số chính là
**giá trị của việc xác nhận vị trí điện cực trước khi triển khai**.

---

## 3. Trong 44 GB, dùng bao nhiêu

| | |
|---|---|
| Toàn bộ CHB-MIT, 24 ca, 23 kênh | **41 GB** (≈ con số 44 GB thường trích) |
| Phải có trên đĩa: 13 ca đánh giá | **25,4 GB** |
| **Thực sự đọc vào model**: 1 kênh, 599,5 h | **1,10 GB** → 2,21 GB khi thành float32 |
| Riêng phần test (hợp của 66 fold, 185 h) | **341 MB** |

**Tỷ lệ byte thực sự đi vào model: 2,5 % của 44 GB.**

Vẫn phải có đủ 25 GB trên đĩa vì file EDF **gói 23 kênh xen kẽ trong cùng một
file** — không tải riêng một kênh được. Chương trình gọi `readSignal(ch_idx)`,
tức vẫn quét qua file nhưng chỉ dựng thành mảng đúng một kênh.

Trong **một fold** còn ít hơn nhiều: chỉ dữ liệu của một bệnh nhân (~46 h, tức
~85 MB một kênh) cộng nhóm 12 người cho giai đoạn tiền huấn luyện.

---

## 4. Cách chia dữ liệu, và vì sao con số của đề tài thấp hơn bài công bố

Điểm phân biệt duy nhất giữa các giao thức là **đơn vị bị tách ra**:

| Cách | Đơn vị tách | Ai dùng |
|---|---|---|
| Mức đoạn, ngẫu nhiên | một cửa sổ 4 s | Chung et al. ở tầng **segment-level** (chia 7:2:1 trên các đoạn chồng lấn) |
| Mức bản ghi | cả file EDF | **đề tài này**; Busia et al. 2025; Chung et al. ở tầng **event-level** |
| Mức bệnh nhân | cả một người | zero-shot (đã cài, chưa chạy) |

Đề tài đã **đo trực tiếp** cái giá của sự khác biệt này. Cùng một kiến trúc, cùng
dữ liệu, chỉ đổi cách chia (66 fold mỗi ô):

| Cách chia | Độ nhạy (mức cửa sổ) | Accuracy | Trùng lặp test/train |
|---|--:|--:|--:|
| Ngẫu nhiên theo cửa sổ | **0,9229** | 0,9968 | **99,6 %** |
| Theo bản ghi | **0,6173** | 0,9887 | 0,0 % |
| Theo bản ghi + bỏ rò rỉ khi khớp ngưỡng | **0,6033** | 0,9888 | 0,0 % |

**Độ nhạy mất 31 điểm phần trăm; accuracy chỉ đổi 0,8 điểm.** Trên bảy ô đã đo,
accuracy trải 1,15 pp còn độ nhạy trải 32 pp.

Nguyên nhân: ở tỷ lệ ictal 0,62 %, một model **không bao giờ báo cơn** đã đạt
**99,38 % accuracy**. Ô tốt nhất trong bảng, 99,68 %, chỉ hơn mức đó 0,30 điểm —
đó là toàn bộ dải động mà accuracy có trên dữ liệu này. **Accuracy không phân
biệt được giao thức có rò rỉ với giao thức sạch.**

*Lưu ý khi đọc bảng:* tỷ lệ ictal khác nhau giữa các dòng (0,62 % ở dòng đầu,
1,42 % ở hai dòng sau), nên accuracy không so trực tiếp giữa các dòng được; độ
nhạy và độ đặc hiệu thì so được.

### Kết quả hiện tại của đề tài

**Độ nhạy mức sự kiện 94,95 %, FAR 0,25/h**, trên 66 fold × 3 seed, phơi nhiễm
**185 h**, 13 ca, 77 cơn.

So với Chung et al. (99,62 %, FAR 0,22/h, ~91 h): con số của đề tài **thấp hơn**,
và khác biệt nằm ở giao thức chứ không nên giấu —

- đề tài giữ lại thêm **một bản ghi không cơn** trong mỗi tập test, để FAR phản
  ánh thời gian bệnh nhân sinh hoạt bình thường chứ không chỉ các file vốn đã có
  cơn (Busia et al. 2025 *chỉ* dùng bản ghi có cơn);
- phơi nhiễm **185 h so với ~91 h**, gấp hơn hai lần;
- ngưỡng hậu xử lý được dò trên tập validation rồi **đóng băng** trước khi chạm
  test;
- lọc thông dải là **nhân quả** (`lfilter`, reset trạng thái mỗi file), không
  dùng `filtfilt` hai chiều vốn làm thông tin tương lai rò về quá khứ.

---

## Tài liệu dẫn

- Y. G. Chung, A. Cho, H. Kim, K. J. Kim, "Single-channel seizure detection with
  clinical confirmation of seizure locations using CHB-MIT dataset,"
  *Frontiers in Neurology*, vol. 15, 1389731, 2024.
  doi:10.3389/fneur.2024.1389731
- P. Busia, G. Leone, A. Matticola, L. Raffo, P. Meloni, "Wearable Epilepsy
  Seizure Detection on FPGA With Spiking Neural Networks," *IEEE TBioCAS*,
  vol. 19, no. 6, pp. 1175–1186, 2025. doi:10.1109/TBCAS.2025.3575327
  — dùng **4 kênh** (F7-T7, T7-P7, F8-T8, T8-P8), leave-one-record-out, chỉ dùng
  bản ghi có cơn, 8 bệnh nhân / 43 cơn / 61 h.
- Số liệu thí nghiệm của đề tài: `docs/EXPERIMENT_LOG_G1a.md` §2e (L5), §2j (thang
  giao thức A7), §2m (định dạng số).
