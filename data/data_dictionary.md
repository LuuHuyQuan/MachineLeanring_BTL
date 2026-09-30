# Data dictionary

Đơn vị quan sát là một ảnh chữ số viết tay đã chuẩn hóa thành 8×8. `load_digits` cung cấp `data` là ma trận 1.797×64, `images` là cách biểu diễn cùng dữ liệu dưới dạng 1.797×8×8 và `target` là 1.797 nhãn. Không xem `images` là nguồn feature bổ sung độc lập.

| Biến | Kiểu và miền | Đơn vị | Vai trò | Thời điểm có sẵn |
|---|---|---|---|---|
| `pixel_0` … `pixel_63` | Số, 0–16 | Mức xám tổng hợp | Feature | Khi có ảnh 8×8 tại thời điểm dự đoán |
| `images[r,c]` | Số, 0–16; `r,c` từ 0–7 | Mức xám | Biểu diễn lại feature | Cùng thời điểm với 64 pixel |
| `target` | Số nguyên, 0–9 | Chữ số | Nhãn học/đánh giá | Chỉ có sau khi gán nhãn; không có khi suy luận |
| Chỉ số hàng | Số nguyên, 0–1796 | Vị trí trong dữ liệu đã nạp | ID kỹ thuật | Khi nạp dữ liệu; không dùng làm feature |
| Chỉ số split | Tập chỉ số hàng train/validation/test | Không có | Metadata đánh giá | Sau khi tách tập, trước huấn luyện |
| `canvas[r,c]` | Số, 0–255; `r,c` từ 0–279 | Mức xám | Đầu vào tương tác | Khi người dùng vẽ; chuyển về feature 8×8 |

`pixel_k` là tên diễn giải trong tài liệu; ma trận NumPy không có tiêu đề cột. Quan hệ vị trí là `k = 8*r + c`, thứ tự từ trái qua phải rồi từ trên xuống dưới. Pixel 0 là nền đen, 16 là mức sáng cao nhất của ảnh Digits; canvas sử dụng thang 0–255 với nền đen và nét trắng.

Schema API dùng key `pixels` cho 64 số hoặc 8×8, key `canvas` cho 280×280. Mỗi request chỉ chọn một key đầu vào. Giá trị boolean, chuỗi, null, không hữu hạn và ngoài miền bị từ chối. Nhãn không được gửi như một feature.

Dữ liệu không có timestamp, ID người viết hoặc thuộc tính nhân khẩu học theo ảnh. Không thể đo sai số theo người viết/giới/tuổi hay khẳng định split độc lập theo người viết từ các trường hiện có. Các nhóm có thể phân tích trực tiếp là chữ số thật 0–9 và cặp nhãn thật/dự đoán.
