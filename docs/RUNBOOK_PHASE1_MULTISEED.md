# Runbook — Phase 1: đường cơ sở có thanh sai số (L7), hai kiến trúc

Mục tiêu của đợt chạy này **không phải nâng điểm**. Nó trả lời hai câu đang chặn mọi việc khác:

1. **Row 22 hay row 24?** Chênh lệch 0.9218 vs 0.9359 là 1.4pp trên 77 cơn — đúng một cơn. Chưa
   con số nào trong 26 run có thanh sai số, nên hiện không thể khẳng định cái nào hơn.
2. **`wearseizure1d` mặc định hay `wearseizure1d_k5only`?** `RESEARCH_REALITY_CHECK.md` §11.4 ghi
   "đã chốt k5only", nhưng toàn bộ rows 21–26 lại chạy kiến trúc mặc định. **L1 + k5only chưa từng
   được thử.** k5only là 11 786 params / 585 920 MACs so với 14 834 / 765 632 — chỉ nó đạt mức
   target của cổng MAC, và luận điểm "ưu thế tính toán 4.3×" của bài báo dựa trên con số của nó.

Commit để checkout: **SHA được bàn giao kèm runbook này** — không hardcode ở đây, vì một SHA viết
sẵn trong file luôn trỏ vào commit *trước* commit chứa chính nó. Lấy nó bằng `git log --oneline -1`
trên máy local sau khi đã review, hoặc từ tin nhắn bàn giao.

---

## 0. Chuẩn bị (một lần)

```bash
cd ~/Manh/WearSeizure-1D
git fetch
git checkout <sha>            # kỷ luật: SHA đã review, không bao giờ đầu nhánh

conda activate chbmit-cnn
export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0
export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts
ulimit -n 65536
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

### 0b. Di chuyển artifacts cũ vào `seed0` — QUAN TRỌNG, làm trước khi chạy bất cứ gì

Artifacts mỗi fold nay nằm sâu thêm một cấp (`.../<window>/seed<N>/`) để hai seed không ghi đè
`*.metrics.json` của nhau. Checkpoint đã sinh ra rows 21–26 đang ở thư mục cha. **Không làm bước
này thì `train.py` sẽ train lại toàn bộ 66 fold từ đầu** và những checkpoint đó thành mồ côi.

```bash
cd "$WEARSEIZURE_ARTIFACTS_DIR"
for d in */patient_specific_loso_edf/*/; do
  if compgen -G "$d"'*.pt' > /dev/null || compgen -G "$d"'*.metrics.json' > /dev/null; then
    mkdir -p "$d/seed0" && mv "$d"*.pt "$d"*.json "$d/seed0"/ 2>/dev/null
    echo "moved -> ${d}seed0"
  fi
done

# Cache init tiền huấn luyện cũng được khoá theo seed.
for d in pretrain/*/*/; do
  if compgen -G "$d"'*.pt' > /dev/null; then
    mkdir -p "$d/seed0" && mv "$d"*.pt "$d"*.json "$d/seed0"/ 2>/dev/null
    echo "moved -> ${d}seed0"
  fi
done
cd ~/Manh/WearSeizure-1D
```

Kiểm tra: `find "$WEARSEIZURE_ARTIFACTS_DIR" -maxdepth 5 -name "seed0" -type d`
— phải thấy cả thư mục model lẫn thư mục pretrain. Nếu quên bước này, `train.py` sẽ in ra đúng
lệnh `mv` cần chạy; đọc cảnh báo thay vì để nó chạy tiếp.

### 0c. Xác nhận trạng thái cũ vẫn tái lập được

Trước khi thêm seed mới, kiểm tra rằng seed 0 đọc đúng những gì đã có và cho lại **đúng row 22**:

```bash
python scripts/evaluate.py profile=server data=chbmit
```

Kỳ vọng: `macro sensitivity=0.922 FAR/h=0.188`, exposure 185.0h. Nếu khác, **dừng lại và báo**,
đừng chạy tiếp — nghĩa là bước di chuyển ở 0b sai hoặc checkpoint không phải cái đã sinh ra row 22.

---

## 1. Ba seed, hai kiến trúc

`train.py` bỏ qua fold nào đã có `metrics.json`, nên seed 0 của `wearseizure1d` sẽ **không** train
lại; chỉ seed 1 và 2 tốn thời gian.

### 1a. Dựng sẵn init tiền huấn luyện (song song, dùng hết máy)

Mỗi seed cần 13 init riêng, vì `cohort_pretrain_fold` lấy tập validation mức bệnh nhân từ
`rng_for(..., base_seed=seed)`. Model chỉ ~12k tham số nên nút cổ chai là **overhead khởi chạy
kernel CUDA**, không phải tính toán — chạy song song nhiều tiến trình hiệu quả hơn nhiều so với
tăng batch size.

```bash
# wearseizure1d, 3 seed x 3 shard. num_workers=4 vì 3 tiến trình chia nhau 14 lõi.
for seed in 0 1 2; do
  for i in 0 1 2; do
    python scripts/pretrain_cohort.py profile=server data=chbmit \
      seed=$seed +shard=$i +n_shards=3 profile.num_workers=4 &
  done
  wait
  echo "=== pretrain inits done for seed $seed (wearseizure1d) ==="
done

# k5only, tương tự
for seed in 0 1 2; do
  for i in 0 1 2; do
    python scripts/pretrain_cohort.py profile=server data=chbmit \
      model=wearseizure1d_k5only seed=$seed +shard=$i +n_shards=3 profile.num_workers=4 &
  done
  wait
  echo "=== pretrain inits done for seed $seed (k5only) ==="
done
```

Lưu ý: seed 0 của `wearseizure1d` sẽ báo *reusing cached init* nếu bước 0b đã chạy đúng. Nếu nó
báo **"cached init ... was built from a different corpus"** ở đợt này thì có gì đó sai — corpus
chưa đổi ở Phase 1; báo lại cho tôi.

### 1b. Huấn luyện

```bash
python scripts/train.py profile=server data=chbmit \
  train.pretrain.enabled=true 'train.seeds=[0,1,2]'

python scripts/train.py profile=server data=chbmit \
  model=wearseizure1d_k5only train.pretrain.enabled=true 'train.seeds=[0,1,2]'
```

### 1c. Bốn cấu hình hậu xử lý

Row 22 và row 24 chỉ khác nhau ở `(run_length, ema_alpha)`, và cả hai đều chỉ là **rethreshold trên
checkpoint đã lưu — không train lại**. Chạy cả hai trên cả hai kiến trúc.

```bash
for model in wearseizure1d wearseizure1d_k5only; do
  for seed in 0 1 2; do
    # --- row 22: run_length=3, ema_alpha=0.125 (mặc định), lưới rộng ---
    python scripts/rethreshold.py profile=server data=chbmit \
      model=$model seed=$seed \
      'postprocess.threshold_search.on_grid=[0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]' \
      'postprocess.threshold_search.off_grid=[0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]'
  done
  echo "=== $model @ row-22 postprocess ==="
  python scripts/evaluate.py profile=server data=chbmit model=$model 'train.seeds=[0,1,2]'
done
```

Chép kết quả về, **rồi** chạy cấu hình row 24 (nó ghi đè cùng các file `metrics.json`):

```bash
for model in wearseizure1d wearseizure1d_k5only; do
  for seed in 0 1 2; do
    # --- row 24: run_length=2, ema_alpha=0.25 ---
    python scripts/rethreshold.py profile=server data=chbmit \
      model=$model seed=$seed \
      postprocess.run_length=2 postprocess.ema_alpha=0.25 \
      'postprocess.threshold_search.on_grid=[0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]' \
      'postprocess.threshold_search.off_grid=[0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]'
  done
  echo "=== $model @ row-24 postprocess ==="
  python scripts/evaluate.py profile=server data=chbmit model=$model \
    postprocess.run_length=2 postprocess.ema_alpha=0.25 'train.seeds=[0,1,2]'
done
```

`evaluate.py` lấy sàn delay từ **tham số đã đóng băng**, không từ config, nên nó báo đúng sàn thật
kể cả khi lệnh `evaluate` thiếu override — nhưng cứ truyền vào cho khớp.

### 1d. Cũng chấm thử theo bộ cổng v2

Không đổi kết quả, chỉ đổi bảng chấm — và nó áp quy tắc worst-patient ≥5 cơn, thứ chưa ai đo:

```bash
python scripts/evaluate.py profile=server data=chbmit \
  eval.gates_path=configs/eval/gates_v2_proposed.yaml 'train.seeds=[0,1,2]'
```

Dòng cần chú ý trong log: `worst-patient sensitivity gated on <bệnh nhân> (<n> events)`. Đó là
**giá trị worst-patient thật sự** của dự án — con số 0.3333 vẫn báo cáo lâu nay là của chb17, bệnh
nhân 3 cơn, thuộc diện miễn trừ.

---

## 2. So sánh, không nhìn điểm số trần trụi

```bash
ART="$WEARSEIZURE_ARTIFACTS_DIR"
BASE="$ART/wearseizure1d/patient_specific_loso_edf/w4s_stride1s"
K5="$ART/wearseizure1d_k5only/patient_specific_loso_edf/w4s_stride1s"

# k5only vs mặc định, gộp cả 3 seed mỗi bên
python scripts/paired_bootstrap.py "$K5" "$BASE" --all-metrics \
  --json "$ART/paired_k5only_vs_default.json"
```

Cách đọc: `sensitivity_macro` có **CI chứa 0** nghĩa là hai kiến trúc không phân biệt được ở độ
phân giải của bộ dữ liệu này — và khi đó **chọn `k5only`**, vì nó rẻ hơn 1.3× về MACs và là biến
thể duy nhất đạt mức target của cổng MAC. Chỉ khi CI **loại trừ 0 và nghiêng về mặc định** thì mới
phải cân nhắc đánh đổi.

Với `far_per_hour_micro` và `delay_mean_s`, **thấp hơn là tốt hơn**, nên delta âm mới có lợi cho A.

---

## 3. Cần gửi về những gì

Với mỗi trong 4 tổ hợp (2 kiến trúc × 2 hậu xử lý):

- Toàn bộ dòng `[3 seeds] <metric>: mean +/- std` từ `evaluate.py`
- Dòng `delay mean=... = floor ... + model reaction ...`
- Dòng `worst-patient sensitivity gated on ...` (từ lần chạy với gate v2)
- Đường dẫn `report_multiseed.json`

Cộng với output của `paired_bootstrap.py`.

Hoặc chỉ cần: `bash scripts/pull_results.sh SERVER-02` rồi báo tôi — nó kéo mọi `*.json` về
`./artifacts/from_server` và bỏ qua checkpoint.

---

## 4. Sau Phase 1 mới tới L5

**Đừng chạy L5 chung với Phase 1.** L1 và L5 phải đo tách nhau, nếu không sẽ không biết cái nào có
tác dụng — đúng như cách nhóm đã làm ở CARE_ASD. Khi Phase 1 xong và kiến trúc đã chốt:

```bash
# Dựng manifest tiền huấn luyện mở rộng (đọc các case CHB-MIT ngoài tập đánh giá).
# Chỉ đọc, không sửa gì; manifest đánh giá 13 case vẫn ghi ra như cũ.
python scripts/make_manifest.py profile=server data=chbmit
```

Đọc kỹ hai dòng log nó in ra: số case ngoài tập đánh giá tìm được, và tổng số giờ tín hiệu. Gửi
cho tôi trước khi chạy tiếp — tôi cần biết corpus thực tế lớn bao nhiêu để ước lượng bộ nhớ, vì
mặc định lấy **cả bốn** vị trí điện cực cho mỗi case.

Cache init sẽ tự động train lại khi corpus đổi (mỗi init mang theo hash của corpus sinh ra nó), nên
không cần xoá tay.

---

## Bẫy đã biết

| Triệu chứng | Nguyên nhân |
|---|---|
| `train.py` train lại cả 66 fold | Chưa chạy bước 0b. Đọc cảnh báo, nó in sẵn lệnh `mv` |
| `Too many open files` | Thiếu `ulimit -n 65536`, hoặc `profile.num_workers` chưa hạ khi chia shard |
| Hydra lỗi khi resolve `hydra.run.dir` | `CHBMIT_RAW_DIR` / `WEARSEIZURE_ARTIFACTS_DIR` chưa set |
| `profile_guard` từ chối chạy | Thiếu `data=chbmit`; `profile=server` một mình sẽ trỏ loader tổng hợp vào dữ liệu lâm sàng |
| `cached init ... different corpus` ở Phase 1 | Không đúng — corpus chưa đổi. Báo lại |
| Sàn delay in ra khác kỳ vọng | `evaluate.py` lấy sàn từ tham số đã đóng băng (đúng), không từ config |
