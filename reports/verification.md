# Biên bản kiểm chứng mã nguồn

Kiểm chứng ngày 30/09/2026 trên Windows với Python 3.12.14 và môi trường trong requirements-lock.txt.

- `python -m pytest -q`: 81 kiểm thử đạt. Các cảnh báo còn lại đến từ tương thích Matplotlib/Pyparsing, không phải lỗi kiểm thử.
- Huấn luyện lại với cùng cấu hình tại thư mục riêng: checksum dữ liệu, split, mô hình MLP và baseline khớp chính xác. Cấu hình chọn và metric validation cũng khớp. Không đánh giá lại test trong bước kiểm chứng này.
- Lịch sử `data/test_access.json` có đúng một lần đánh giá test hoàn tất.
- GET trang giới thiệu, nhận dạng, dashboard, metadata, demo, health và các tài nguyên giao diện/biểu đồ đều trả HTTP 200.
- Cả 10 mẫu demo lấy từ train đều nhận đủ 10 xác suất với tổng bằng 1. Đây chỉ là kiểm tra tích hợp, không phải đánh giá tổng quát hóa.
- Trình duyệt: canvas rỗng được thông báo lỗi; mẫu train và nét vẽ chuột đi qua API; ảnh 8×8 và xác suất hiển thị đúng. Một nét số 1 vẽ trong lần kiểm tra được dự đoán là 1.
- Dashboard hiển thị MLP accuracy 96,11%, macro-F1 0,9607, log loss 0,1014; baseline accuracy 95,56%; tập test 360 ảnh.
- Kiểm tra viewport 390×844: trang nhận dạng và dashboard không tràn ngang; các hình trên dashboard tải thành công.
- Không ghi nhận lỗi JavaScript trong console ở lần kiểm tra cuối.

Ảnh dashboard được lưu tại `docs/screenshots/dashboard.jpg`. Kết quả kiểm chứng này không bảo đảm mọi nét chữ ngoài bộ Digits được nhận dạng đúng.
