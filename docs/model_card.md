# Model card

## Mục đích

Ứng dụng học tập Project 28, dự đoán một chữ số 0–9 từ ảnh 8×8. Model chính là MLPClassifier, so sánh với LogisticRegression. Toàn bộ Pipeline gồm scaling hằng16 và classifier được lưu bằng joblib để serving không huấn luyện lại.

## Artifact và cấu hình

Model phục vụ: `models/digit_pipeline.joblib`. Baseline: `models/baseline_pipeline.joblib`. Metadata: `models/model_metadata.json`. Đọc metadata thực tế để biết cấu hình được chọn, seed, thư viện, thời điểm và tiêu chí lựa chọn; tài liệu này không dự đoán trước kết quả huấn luyện.

## Dữ liệu và đánh giá

Nguồn: [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), 1.797 ảnh 8×8, miền pixel 0–16, nhãn 0–9. Chia phân tầng 60/20/20 với seed 42. Validation dùng chọn model; test giữ riêng cho báo cáo cuối; learning curve 3 fold chỉ trong train.

Xem `reports/experiments.json` cho validation, `reports/evaluation.json` cho test và `reports/figures/` cho confusion matrix, learning curve, lịch sử huấn luyện và ảnh lỗi. Khi chưa đánh giá, metric test chưa có; không suy ra số test từ validation. Nếu MLP không vượt baseline, cần báo trung thực và giải thích từ độ phức tạp, regularization, hội tụ và dữ liệu nhỏ.

## Đầu vào, đầu ra và người sử dụng

Đầu vào hợp lệ là 64 pixel hoặc 8×8 trong 0–16, hoặc canvas 280×280 trong 0–255 với nét trắng/nền đen. API kiểm tra số hữu hạn, kiểu, shape, miền và ảnh rỗng. Đầu ra gồm nhãn, xác suất 10 lớp, độ tin cậy, ảnh 8×8 và cảnh báo.

Người dùng mục tiêu là sinh viên/người học mạng neuron. Model không nhận dạng một dãy chữ số, trang văn bản hoặc nội dung tài liệu pháp lý. Không dùng cho quyết định ảnh hưởng quyền lợi con người.

## Giới hạn và nguy cơ diễn giải

- Dữ liệu nhỏ, độ phân giải thấp; điểm trên Digits không đại diện mọi chữ viết.
- Canvas người dùng có thể lệch vị trí, tỷ lệ và độ dày nét so với train.
- Không có ID người viết để đánh giá độc lập theo người viết hoặc nhân khẩu học.
- Confidence là xác suất classifier, chưa hiệu chỉnh riêng và không bảo đảm đúng.
- Mô hình có thể đoán rất tự tin trên ảnh ngoài miền. Cảnh báo giao diện không thay thế đánh giá ngoài miền.

## Duy trì và cập nhật

Giữ nguồn, checksum, split, requirements và metadata. Chỉ nạp model từ nguồn tin cậy. Trước cập nhật, đóng băng thiết kế thí nghiệm mới và xác định dữ liệu đánh giá độc lập; không liên tục tối ưu trên test đã xem. Lưu lý do và dấu vết nếu tái chạy kỹ thuật bằng `--force --reason`.
