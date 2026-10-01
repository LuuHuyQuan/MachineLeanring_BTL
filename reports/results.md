# Kết quả chạy thực tế

Thời điểm đánh giá (UTC): 2026-09-30T13:09:02+00:00.

Split stratified seed 42: {'train': 1077, 'validation': 360, 'test': 360}. Test được giữ riêng khi chọn cấu hình.

| Mô hình | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|
| Logistic Regression | 0.9556 | 0.9546 | 0.1758 |
| MLP | 0.9611 | 0.9607 | 0.1014 |

MLP chênh +0.56 điểm phần trăm accuracy so với baseline trên cùng 360 ảnh test.
Đây là kết quả một split nhỏ, không chứng minh MLP luôn tốt hơn baseline trên dữ liệu khác.

Cấu hình chọn bằng validation: `mlp_128_64_alpha` — `{'hidden_layer_sizes': [128, 64], 'alpha': 0.01, 'early_stopping': True}`.
Xem reports/experiments.csv cho so sánh baseline/MLP, một/hai lớp ẩn và alpha/early stopping; evaluation.json chứa phân tích lỗi.

Cặp nhầm nhiều nhất: 1 ↔ 8, tổng 7 lỗi hai chiều.
Tổng 14 ảnh MLP dự đoán sai được lưu trong evaluation.json; ví dụ ở figures/error_examples.png.
Không sửa siêu tham số hoặc bộ chuyển canvas dựa trên các lỗi test này.

Learning curve: trung bình và độ lệch chuẩn 3 fold chỉ trên TRAIN; không phải khoảng tin cậy của test.
Phân tích hình ảnh có thể gợi ý nét mờ/biến thể hình dạng; không đủ dữ liệu để kết luận nguyên nhân lỗi.

Nguồn dữ liệu: [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
[UCI Optical Digits](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits).
