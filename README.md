# Nhận dạng chữ số viết tay bằng MLP

Project 28 — Học máy cơ bản. Ứng dụng Flask cho phép vẽ chữ số, xem ảnh 8×8 sau tiền xử lý, xem xác suất 10 lớp và đọc kết quả đánh giá. Dữ liệu là 1.797 ảnh trong `scikit-learn Digits`; mô hình chính là `MLPClassifier`, baseline là `LogisticRegression`.

## Chạy từ máy mới

Cài Python 3.12, mở terminal tại thư mục dự án rồi thực hiện các lệnh sau. Bản bàn giao có model đã huấn luyện và kết quả thật, nên có thể mở web ngay sau khi cài thư viện. Cần Internet để cài thư viện lần đầu; dữ liệu Digits được đóng gói trong scikit-learn, không cần tài khoản hay tải MNIST.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m app
```

Trên macOS/Linux, thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`. Khi đã kích hoạt môi trường, có thể dùng trực tiếp `python`. Mở [http://127.0.0.1:5000](http://127.0.0.1:5000). Giữ terminal chạy ứng dụng; nhấn `Ctrl+C` để dừng.

Những lần mở web sau chỉ cần `python -m app`: ứng dụng nạp Pipeline đã lưu, không huấn luyện trong request. Nếu thiếu model, API trả `503` và hướng dẫn chạy bước huấn luyện. `requirements.txt` cố định các thư viện chính; `requirements-lock.txt` cố định cả các phụ thuộc gián tiếp trong môi trường đã kiểm chứng.

Trên Windows có thể chạy `.\run.bat` hoặc `powershell -ExecutionPolicy Bypass -File .\launch.ps1`. Script tạo môi trường/cài thư viện khi cần rồi mở server. Thêm `-Check` vào cuối lệnh PowerShell để chạy kiểm thử; thêm `-Reproduce` để tái lập trong thư mục riêng. Các lệnh Python ở trên là cách chạy minh bạch.

## Tái lập từ dữ liệu đến kết quả

Để không ghi đè kết quả bàn giao đã đóng băng, dùng một thư mục artifact mới. Sau khi đã kích hoạt môi trường Python:

```powershell
python -m src.data --root runs/reproduction
python -m src.train --root runs/reproduction
python -m src.evaluate --root runs/reproduction
python -m app --root runs/reproduction
```

Các script tạo thư mục khi cần. Nếu `runs/reproduction` đã có kết quả test, dùng một tên mới như `runs/reproduction_2` cho mọi lệnh. Một lần chạy lại để kiểm chứng kỹ thuật không phải một test mới chưa từng được xem. Không dùng khác biệt giữa các lần tái lập để chọn seed hoặc chỉnh model. Huấn luyện và learning curve cần thời gian; mở web chỉ cần model có sẵn.

## Quy trình và tính độc lập của test

1. `python -m src.data`: nạp Digits, ghi nguồn và checksum, kiểm tra chất lượng, tạo split có phân tầng theo nhãn và EDA trên train.
2. `python -m src.train`: so sánh baseline với số cấu hình MLP có giới hạn; chọn theo validation; tạo learning curve bằng 3 fold chỉ trong train; lưu model và cấu hình đã chọn.
3. `python -m src.evaluate`: đánh giá model đã đóng băng trên test, ghi metric, ma trận nhầm lẫn và ví dụ lỗi.
4. `python -m app`: phục vụ model có sẵn và hiển thị kết quả trên dashboard.

Split cố định `random_state=42`, tỷ lệ xấp xỉ `60% / 20% / 20%` cho train/validation/test; sai lệch tối đa do số mẫu nguyên. Hai mô hình dùng cùng split. Pixel được chia cho hằng `16` trong `Pipeline`, đưa miền `[0,16]` về `[0,1]`; phép chia này không ước lượng cực trị trên toàn bộ dữ liệu.

Validation dùng để chọn cấu hình. `early_stopping=True` của MLP còn tách một phần dữ liệu **train** làm validation nội bộ để quyết định dừng. Phần này khác validation bên ngoài dùng so sánh các cấu hình. Không dùng test để chọn lớp ẩn, alpha, epoch, seed hoặc chỉnh thuật toán canvas.

Test chỉ dùng một lần cho kết luận cuối. Khi đã có kết quả test, script đánh giá và huấn luyện từ chối ghi đè mặc định. Khi cần khôi phục kỹ thuật, dùng `--force --reason "Lý do cụ thể"`; script lưu dấu vết và bản lưu của kết quả cũ. Tùy chọn này không phải cách thử nhiều cấu hình trên test. Sau khi xem test, mọi thay đổi dựa vào lỗi test cần được công bố là phân tích khám phá, không coi kết quả của lần đánh giá lại là một test độc lập mới.

Các chỉ số được báo cáo gồm accuracy, macro-F1, confusion matrix và log loss. Learning curve kèm biến thiên giữa các fold chỉ phản ánh dữ liệu train. Dashboard tách kết quả validation và test; không điền số liệu khi chưa chạy đánh giá.

## Sử dụng web và API

Web có ba màn hình: giới thiệu/phạm vi, vẽ và nhận dạng, dashboard đánh giá/model card. Vẽ nét trắng trên nền đen; nút xóa tạo lại canvas. Kết quả gồm nhãn, độ tin cậy, xác suất từng chữ số và ảnh 8×8 mà model thực sự nhận.

`POST /api/digit` nhận JSON có **một** chế độ đầu vào:

- `pixels`: danh sách 64 số hoặc ma trận 8×8; mỗi số trong `[0,16]`.
- `canvas`: ma trận 280×280 số, nét trắng nền đen; mỗi số trong `[0,255]`.

Mọi giá trị phải là số hữu hạn. API từ chối chuỗi, boolean, NaN/Infinity, kích thước sai, dữ liệu rỗng hoặc ngoài miền; không tự sửa dữ liệu không hợp lệ. Chi tiết request/response, mã lỗi và ví dụ PowerShell ở [docs/api.md](docs/api.md).

## Cấu trúc dự án

```text
app/                    Flask, HTML/CSS/JavaScript và API
src/data.py             Dữ liệu, kiểm tra chất lượng, split, EDA train
src/features.py         Validation và tiền xử lý pixel/canvas
src/train.py            Baseline, MLP, chọn model và learning curve
src/evaluate.py         Đánh giá test cuối và phân tích lỗi
data/                   Data card, dictionary và cache có thể tái tạo
models/                 Pipeline và metadata sau khi huấn luyện
reports/                Kết quả và biểu đồ sinh từ pipeline
docs/                   API, phương pháp và tài liệu bàn giao
tests/                  Kiểm thử dữ liệu, tiền xử lý và API
requirements.txt        Phiên bản thư viện cố định
launch.ps1              Hỗ trợ chạy trên Windows
```

Đường dẫn trong code tính từ thư mục dự án; không cần sửa đường dẫn cá nhân. Dữ liệu cache và các lần tái lập riêng không đưa vào Git, có thể sinh lại bằng pipeline. Model nhị phân nhỏ được kèm theo để web chạy ngay. Không nạp file joblib do người lạ cung cấp vì việc deserialize model cần nguồn tin cậy.

Các artifact chính:

- `data/manifest.json`, `data/splits.json`, `data/demo_samples.json` và `data/raw/digits.npz`.
- `models/digit_pipeline.joblib`, `models/baseline_pipeline.joblib` và `models/model_metadata.json`.
- `reports/data_quality.json`, `reports/experiments.json`, `reports/learning_curve.json`, `reports/evaluation.json`.
- `reports/figures/`: phân bố lớp, ảnh train, learning curve, lịch sử huấn luyện, confusion matrix và ví dụ lỗi.

Kết quả test của bản bàn giao trên 360 ảnh: MLP đạt accuracy **96,11%**, macro-F1 **0,9607**, log loss **0,1014**; baseline đạt **95,56%**, **0,9546**, **0,1758** tương ứng. Chênh lệch accuracy khoảng **0,56 điểm phần trăm** trên split này, chưa chứng minh MLP vượt trội trên mọi dữ liệu. Xem [reports/results.md](reports/results.md) và JSON để biết cấu hình và phân tích lỗi thực tế.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Kiểm thử tập trung vào split không chồng lấn và có phân tầng, scaler cố định, validation đầu vào, chuyển canvas và hợp đồng API. Kiểm thử không dùng test set để chọn model. Để xác minh end-to-end, chạy đủ bốn lệnh ở phần **Tái lập từ dữ liệu đến kết quả** trong môi trường sạch, sau đó thử một ảnh mẫu, vẽ một chữ số và gửi một đầu vào sai tới API.

## Dữ liệu, giới hạn và tài liệu bàn giao

Đọc [data/README.md](data/README.md), [data/data_dictionary.md](data/data_dictionary.md), [docs/methodology.md](docs/methodology.md) và [docs/model_card.md](docs/model_card.md). Digits gồm ảnh nhỏ đã chuẩn hóa. Chữ vẽ bằng chuột có thể khác phân bố train; confidence là xác suất model đưa ra, không bảo đảm dự đoán đúng và chưa được hiệu chỉnh xác suất riêng. Không dùng ứng dụng này cho tài liệu pháp lý, nhận dạng quan trọng hay nhận dạng một trang văn bản.

Mã nguồn và tài liệu trong dự án hỗ trợ hoàn thiện bài nộp. [docs/report_outline.md](docs/report_outline.md) là đề cương để viết báo cáo 15–25 trang; [docs/presentation_outline.md](docs/presentation_outline.md) là đề cương 10–12 slide; [docs/demo.md](docs/demo.md) là kịch bản demo 5–7 phút. Các đề cương này không thay thế báo cáo DOCX/PDF hoặc file trình chiếu hoàn chỉnh.

[docs/team_plan.md](docs/team_plan.md) là mẫu phân công và nhật ký 6 tuần. Điền tên, giờ làm, kết quả và commit thực tế của hai thành viên. Không tạo Git history hay đóng góp giả. Ghi rõ phần có trợ giúp AI và cách nhóm kiểm chứng trước khi nộp.

## Nguồn

- [scikit-learn: load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [UCI: Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits).
- [scikit-learn: neural networks supervised](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).
- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
