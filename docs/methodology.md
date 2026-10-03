# Phương pháp và thiết kế thí nghiệm

## Bài toán và thời điểm dự đoán

Phân loại một ảnh chữ số viết tay 8×8 thành một trong 10 nhãn, trả xác suất và nhãn. Tại thời điểm dự đoán, chỉ pixel có sẵn. Nhãn thật và kết quả đánh giá không phải feature. Câu hỏi: MLP thay đổi accuracy/macro-F1 so với logistic regression bao nhiêu trên cùng split, và số lớp ẩn, alpha, early stopping ảnh hưởng khả năng tổng quát thế nào?

## Split, preprocessing và leakage

Chia phân tầng theo nhãn bằng seed 42 thành train/validation/test xấp xỉ 60/20/20 trước các bước học. Chỉ số split được lưu. Kiểm tra schema toàn dữ liệu là kiểm tra toàn vẹn, không được dùng nhãn test hoặc lỗi test để chỉnh model. EDA phục vụ lựa chọn thực hiện trên train.

`PixelScaler` trong Pipeline chia cho hằng 16 đã biết từ định nghĩa dataset. Khác scaler học min/max từ dữ liệu, phép biến đổi này không fit một thống kê từ test. Transformer hoạt động với dữ liệu dạng số 64 feature; validation request web có kiểm tra bổ sung kiểu dữ liệu, miền và nét vẽ. Cả baseline lẫn MLP dùng cùng phép chia.

Canvas trắng trên nền đen được kiểm tra rồi crop vùng nét, giữ tỷ lệ khi resize vào vùng 210 pixel, căn giữa trên nền 280×280, lấy trung bình từng ô 35×35 để tạo 8×8 và nhân 16/255. Mọi chế độ suy luận cuối cùng đều vào cùng Pipeline 64 feature. Cần phân biệt điều này với giả định ảnh chuột vẽ có cùng phân bố ảnh Digits: cùng dạng input không làm mất sai khác miền. Hình 8×8 giúp kiểm tra nét bị méo hoặc quá mờ.

## Baseline và MLP

Baseline là `LogisticRegression`, ranh giới quyết định theo tổ hợp tuyến tính của các pixel sau scaling. MLP thêm một hoặc hai lớp ẩn ReLU để học biểu diễn phi tuyến; regularization L2 điều khiển bằng `alpha`. Giữ giới hạn số cấu hình giúp thí nghiệm có thể chạy trên máy cá nhân và giảm việc lựa chọn tùy tiện.

Các cấu hình cụ thể, số neuron, `max_iter`, alpha, early stopping và seed phải lấy từ `reports/experiments.json` và `models/model_metadata.json`, không chép số dự kiến vào bảng kết quả. Cùng validation ngoài dùng so sánh mọi cấu hình.

Với early stopping, scikit-learn tách một phần train làm validation nội bộ và theo dõi tiêu chí dừng. Phần nội bộ chỉ phục vụ dừng huấn luyện, không thay validation ngoài dùng chọn cấu hình. Khi không early stopping, tối ưu dừng theo tiêu chí hội tụ hoặc `max_iter`; warning không hội tụ cần ghi nhận thay vì giấu.

## Bốn thí nghiệm bắt buộc

| Thí nghiệm | Cách so sánh | Bằng chứng |
|---|---|---|
| Baseline và MLP | Cùng split, cùng scaler, cùng metric | Bảng validation và bảng test cuối tách riêng |
| Một và hai lớp ẩn | Kiến trúc được giới hạn, đối chiếu cấu hình | Bảng cấu hình, metric và thời gian huấn luyện |
| Alpha và early stopping | Đối chiếu các cấu hình kiểm soát đã định trước | Metric validation, số vòng lặp và lịch sử huấn luyện |
| Cặp chữ số nhầm nhiều nhất | Phân tích confusion matrix sau đánh giá cuối | Cặp nhầm, số lượng và ảnh lỗi cụ thể |

Không diễn giải thay đổi alpha/kiến trúc là quan hệ nhân quả trên mọi dataset. Nếu đồng thời thay nhiều tham số, ghi rõ đây là so sánh cấu hình chứ không tách được hiệu ứng riêng từng tham số.

## Lựa chọn và đánh giá

Chỉ chọn model từ validation: ưu tiên accuracy cao hơn, sau đó macro-F1 cao hơn, cuối cùng log loss thấp hơn khi hòa. Chỉ cấu hình MLP có `early_stopping=True` đủ điều kiện chọn; cấu hình không early stopping dùng cho thí nghiệm đối chiếu. Model được đóng băng cùng metadata trước đánh giá test; không refit âm thầm trên validation sau lựa chọn. Learning curve chạy 3 fold trong train; mỗi fold fit Pipeline trên phần train fold. Điểm fold và độ lệch chuẩn không phải độ bất định của kết quả test.

Accuracy đo tỷ lệ nhãn đúng. Macro-F1 tính trung bình F1 mỗi lớp với trọng số bằng nhau. Log loss đo chất lượng phân bố xác suất; sai rất tự tin bị phạt mạnh. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán. Luôn đọc số mẫu kèm metric.

`src.evaluate` dùng test một lần sau đóng băng và phân tích lỗi. Không dùng ảnh lỗi test để chỉnh tham số hoặc preprocessing rồi công bố cùng test là đánh giá độc lập. Cơ chế `--force --reason` phục vụ tái chạy kỹ thuật có lưu dấu vết; không thể biến test đã xem thành dữ liệu chưa từng xem.

## Giới hạn

Chỉ có 1.797 ảnh nhỏ, nguồn từ chữ số chuẩn hóa và không có ID người viết từng ảnh trong loader. Split theo ảnh không chứng minh độc lập theo người viết. Không có đánh giá trên dữ liệu chữ vẽ chuột được gán nhãn riêng; demo web không phải nghiên cứu độ chính xác ngoài miền. Xác suất chưa hiệu chỉnh riêng; một chữ số khác miền vẫn có thể nhận dự đoán rất tự tin.

Nguồn phương pháp: [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [neural networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html), [data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
