# Đề cương báo cáo 15–25 trang

Đây là đề cương để viết báo cáo, không phải báo cáo hoàn chỉnh đã đạt số trang. Tạo DOCX hoặc LaTeX và xuất PDF sau khi kiểm chứng kết quả. Số trang dưới đây là gợi ý cho nội dung chính, không tính phụ lục.

| Mục | Nội dung cần viết và minh chứng | Trang gợi ý |
|---|---|---|
| Tóm tắt | Bài toán, dữ liệu, phương pháp, kết quả thực tế và giới hạn chính | 1 |
| Bối cảnh và câu hỏi | Đơn vị ảnh, thời điểm dự đoán, đầu ra và tiêu chí thành công | 1–2 |
| Dữ liệu và giấy phép | Nguồn UCI/scikit-learn, 1.797 ảnh, schema, checksum, quyền sử dụng | 2 |
| Chất lượng và EDA | Missing, trùng, ngoại lệ, phân bố lớp train, ảnh mẫu, quyết định từ EDA | 2 |
| Split và leakage | Stratified 60/20/20, seed 42, scaler cố định, validation nội bộ và ngoài | 1–2 |
| Phương pháp | LogisticRegression, MLP 1–2 lớp ẩn, ReLU, alpha và early stopping | 2–3 |
| Thiết kế thí nghiệm | Bốn thí nghiệm, cấu hình hữu hạn, quy tắc chọn và learning curve train-only | 1–2 |
| Kết quả | Bảng validation riêng, bảng test riêng, accuracy/macro-F1/log loss, learning curve | 2 |
| Phân tích lỗi | Confusion matrix, cặp nhầm nhiều nhất, ảnh lỗi và giải thích có giới hạn | 1–2 |
| Web và API | Ba màn hình, request/response, validation và luồng serving model đã lưu | 1–2 |
| Đạo đức và giới hạn | Ngoài miền, confidence, dữ liệu nhỏ, không dùng cho tài liệu pháp lý | 1 |
| Kết luận | Trả lời câu hỏi từ kết quả thực tế; không phóng đại; hướng cải thiện | 1 |
| Tài liệu tham khảo | Tài liệu học phần, UCI, scikit-learn và nguồn thực sự đã sử dụng | 1 |

## Các bảng và hình nên có

1. Data dictionary và bảng kiểm tra chất lượng.
2. Phân bố lớp train và ảnh mẫu train.
3. Bảng cấu hình baseline/MLP: hidden layers, alpha, early stopping, seed, số vòng lặp, thời gian.
4. Bảng metric validation dùng chọn model, bảng metric test dùng kết luận đặt riêng.
5. Learning curve 3 fold train-only với số mẫu và biến thiên giữa fold.
6. Confusion matrix với nhãn hàng/cột và ví dụ cặp chữ số nhầm nhiều nhất.
7. Sơ đồ luồng offline data→train→evaluate và luồng request→preprocess→Pipeline→JSON.
8. Ảnh chụp ba màn hình web và ví dụ lỗi API.

Lấy số liệu từ `reports/*.json`; lấy hình từ `reports/figures/`. Mỗi hình cần chú thích, đơn vị, nguồn và giải thích tác động tới kết luận. Không điền các con số minh họa thành kết quả thực nghiệm.

## Phụ lục tái lập

- Phiên bản Python/thư viện, lệnh cài đặt và lệnh chạy.
- Config, seed, checksum, kích thước split và quy tắc chọn model.
- Hợp đồng API và cách chạy kiểm thử.
- Link repo, commit/release đã nộp, bảng phân công, nhật ký thật.
- Khai báo công cụ AI: phần được hỗ trợ, phần nhóm tự thực hiện và cách kiểm chứng.
- Mọi lần đánh giá lại test với lý do, nếu có; phân biệt đánh giá kỹ thuật lại với kết quả độc lập.

## Kiểm tra trước xuất PDF

Đối chiếu số liệu với artifact, kiểm tra nguồn cho mọi bảng/hình, bảo đảm hình đọc được và không vượt lề. Báo cáo trung thực nếu MLP không vượt baseline. Tổng trang thực tế phải đáp ứng đề bài; không dùng phụ lục hoặc ảnh phóng to để thay nội dung phân tích.
