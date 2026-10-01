# Model card

Mục đích: học tập nhận dạng một chữ số viết tay 0–9, ảnh xám 8×8.

Mô hình: MLPClassifier `mlp_128_64_alpha`; `{'hidden_layer_sizes': [128, 64], 'alpha': 0.01, 'early_stopping': True}`.
Dữ liệu: 1.797 ảnh Digits; CC BY4.0 (UCI); seed 42; sklearn 1.7.2.
Tiền xử lý: 8×8 grayscale pixels0..16; PixelScaler divides by16 in both training and API.
Canvas: White strokes on black280×280; crop, aspect-preserving center, area mean to8×8, intensity×16/255.

Kết quả test: Accuracy 0.9611, macro-F1 0.9607, log loss 0.1014.

Giới hạn:

- Dữ liệu nhỏ:1.797 ảnh chuẩn hóa8×8; độ chính xác test không đại diện mọi nét vẽ canvas.
- Canvas dùng bộ chuyển ảnh cố định; vẫn có khác biệt miền dữ liệu với Digits gốc.
- Chỉ nhận một chữ số0–9; không hỗ trợ chữ, nhiều chữ số, ảnh tài liệu hay mục đích pháp lý.
- Xác suất chưa hiệu chỉnh; độ tin cậy cao không bảo đảm đầu vào nằm trong miền.
- Không có ID người viết để kiểm chứng tổng quát hóa theo người viết độc lập.

API từ chối dữ liệu sai schema/ngoài miền; không phát hiện đầy đủ đầu vào ngoài miền (ví dụ chữ cái).
Không thu thập hoặc lưu nét vẽ của người dùng; ảnh được xử lý trong bộ nhớ phục vụ request.

SHA256 model: `6ff9541910af248e07efac20cfeb33763a972e68f6e65ad058a6f44880c55154`.
SHA256 dữ liệu: `f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443`.
