# Kịch bản demo trực tiếp 5–7 phút

## Chuẩn bị trước buổi demo

Chạy dữ liệu, huấn luyện, đánh giá test đúng một lần và mở ứng dụng. Kiểm tra ba màn hình, model card, biểu đồ và API. Lưu sẵn một ảnh mẫu hợp lệ từ demo trong app. Không chạy lại đánh giá test chỉ để trình diễn. Đóng tab/cửa sổ chứa thông tin cá nhân và dùng dữ liệu an toàn.

## Luồng đề xuất

| Thời lượng | Thao tác | Nội dung giải thích |
|---|---|---|
| 0:00–0:45 | Mở màn hình giới thiệu | Bài toán một chữ số 0–9, nguồn Digits, ảnh 8×8, mục đích học tập |
| 0:45–2:15 | Mở màn hình vẽ, nhập chữ số, nhận dạng | Canvas trắng/nền đen; xem ảnh 8×8 mà model nhận; nhãn và xác suất 10 lớp |
| 2:15–2:45 | Xóa và thử lại hoặc dùng một ảnh mẫu | Cho thấy nút xóa; phân biệt ảnh mẫu train/demo với chất lượng chữ vẽ chuột |
| 2:45–3:30 | Gửi pixel sai kích thước hoặc canvas rỗng | API trả lỗi có nghĩa, UI không sập; không tự thay thế giá trị sai |
| 3:30–5:00 | Mở dashboard | Tách validation/test; baseline và MLP; kiến trúc, alpha, early stopping; đọc confusion matrix và ảnh lỗi |
| 5:00–6:00 | Mở model card và giải thích pipeline | Scaling hằng 16, split seed 42, learning curve train-only; model đã lưu được phục vụ offline |
| 6:00–7:00 | Nêu giới hạn, kết thúc demo | Confidence không bảo đảm đúng; canvas ngoài miền; test không dùng để chọn model |

Số liệu trình bày phải lấy từ dashboard/artifact đã chạy thực tế. Nếu ảnh vẽ sai, giải thích hình 8×8 và phân bố xác suất; không giấu trường hợp lỗi. Nếu model chưa huấn luyện, trình bày rõ lỗi `503` rồi dùng artifact hợp lệ đã chuẩn bị; không bịa metric.

## Phân công hai người

Thành viên A: giới thiệu, thao tác vẽ, giải thích preprocessing. Thành viên B: dashboard, thí nghiệm và giới hạn. Đây là đề xuất; điền tên thực tế vào nhật ký. Cả hai cần thực hành đổi vai để trả lời vấn đáp về toàn pipeline.

## Câu hỏi cần chuẩn bị

- Vì sao MLP có thể học quan hệ phi tuyến mà logistic regression không biểu diễn trực tiếp?
- Validation ngoài và validation nội bộ early stopping khác nhau thế nào?
- Vì sao learning curve không dùng test?
- Vì sao cùng shape 8×8 vẫn chưa bảo đảm chữ vẽ chuột cùng miền với Digits?
- Macro-F1 và log loss bổ sung gì cho accuracy?
- Sau khi xem lỗi test, có thể tối ưu tiếp và vẫn gọi test độc lập không?
