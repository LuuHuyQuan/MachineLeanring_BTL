# Hồ sơ dữ liệu — scikit-learn Digits

## Nguồn và quyền sử dụng

Dự án dùng `sklearn.datasets.load_digits()` với cấu hình mặc định: 1.797 ảnh xám 8×8, nhãn chữ số 0–9 và 64 đặc trưng pixel có giá trị từ 0 đến 16. Đây là bản dữ liệu Digits được phân phối trong scikit-learn, dựa trên tập test của UCI Optical Recognition of Handwritten Digits; không phải MNIST.

Nguồn chính thức:

- [scikit-learn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits), DOI `10.24432/C50P49`.

Trang UCI công bố giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Khi phân phối dữ liệu hoặc viết báo cáo cần giữ nguồn, trích dẫn và nêu các biến đổi đã thực hiện. Code scikit-learn có giấy phép riêng; giấy phép đó không thay thế việc ghi nguồn dataset.

Trích dẫn dữ liệu đề xuất: Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.

## Tái tạo và kiểm chứng

```powershell
python -m src.data
```

Lệnh nạp dữ liệu đóng gói sẵn, tạo cache, ghi metadata nguồn/phiên bản/checksum, kiểm tra chất lượng, sinh split và biểu đồ EDA train. Không cần tải thủ công file từ UCI. Ngày lấy dữ liệu là thời gian thực thi được ghi trong metadata; không điền ngày tải giả vào tài liệu này. Phiên bản scikit-learn có trong `requirements.txt` và metadata của pipeline. Checksum được tính từ dữ liệu thực tế giúp kiểm chứng dữ liệu giữa các lần chạy.

Thông tin bản bàn giao theo `manifest.json`: nạp ngày **30/09/2026**, scikit-learn **1.7.2**, shape **1.797×64**. SHA-256 của dữ liệu theo định dạng `float64 little-endian X + int64 little-endian y` là:

```text
f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443
```

Các file dữ liệu sinh trong `data/raw/` và `data/processed/` bị bỏ qua bởi Git. Các file tài liệu, manifest, split và demo an toàn được quản lý phiên bản. Để tái lập, dùng `requirements-lock.txt`, seed 42 và các split đã sinh từ script; không thay đổi split sau khi xem test. Bản bàn giao có kết quả cuối, nên tái lập vào thư mục riêng bằng `python -m src.data --root runs/reproduction`, rồi chạy train/evaluate với cùng `--root` như hướng dẫn README.

## Chất lượng và xử lý

Báo cáo chất lượng của script kiểm tra số hàng, shape, miền pixel, giá trị thiếu/không hữu hạn, nhãn, hàng trùng và phân bố lớp. Danh sách hàng bị loại và lý do phải được ghi nếu có loại bỏ; không tự xóa ảnh khó chỉ vì model dự đoán sai. Ảnh trùng được báo cáo để biết chất lượng dữ liệu; quyết định xử lý cần được ghi trước khi đánh giá cuối.

Bản bàn giao có **0** giá trị thiếu, **0** ảnh trùng, **0** pixel ngoài miền và **0** hàng bị loại. Pixel thực tế từ 0 đến 16. Split gồm **1.077 train, 360 validation, 360 test**. Chi tiết phân bố 10 lớp và quyết định xử lý nằm trong `reports/data_quality.json`; biểu đồ phục vụ lựa chọn chỉ sử dụng train.

Đặc trưng là 64 pixel, không thêm thông tin nhãn vào đầu vào. Không fit imputer, scaler theo min/max của toàn bộ dữ liệu, PCA hoặc chọn feature trên test. Script tiền xử lý dùng phép chia cố định cho 16; model học từ train trong Pipeline. EDA phục vụ quyết định mô hình dùng train.

Split phân tầng theo nhãn, `random_state=42`, tỷ lệ xấp xỉ 60/20/20. Cơ chế này không bảo đảm tách theo người viết: dữ liệu loader không cung cấp ID người viết cho từng ảnh. Vì vậy kết quả đánh giá phản ánh việc chia các ảnh của bộ Digits, không chứng minh khả năng tổng quát sang người viết chưa từng gặp.

## Phạm vi và biến đổi khi dự đoán

Một ảnh đầu vào API có thể đã ở miền 8×8/0–16 hoặc là canvas 280×280/0–255. Canvas được kiểm tra, đưa về 8×8 bằng quy trình trong `src/features.py`, rồi đi qua cùng Pipeline chia 16 và phân loại. Giảm kích thước canvas tạo thêm sai khác miền so với ảnh Digits gốc; cần xem ảnh 8×8 hiển thị trước khi diễn giải kết quả.

Xem [data_dictionary.md](data_dictionary.md) cho schema. Dataset phục vụ học tập và thử nghiệm; không có đảm bảo dùng cho chữ viết trong tài liệu thực tế.
