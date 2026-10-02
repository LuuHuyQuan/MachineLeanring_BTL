# API nhận dạng chữ số

Ứng dụng mặc định chạy tại `http://127.0.0.1:5000`. Model được nạp từ artifact đã huấn luyện; API không học lại từ request.

## `POST /api/digit`

Header: `Content-Type: application/json`. Body là object có **một** trường đầu vào `pixels` hoặc `canvas`.

| Trường | Kích thước | Miền | Quy ước |
|---|---|---|---|
| `pixels` | 64 phần tử hoặc 8×8 | 0–16 | Nền đen, nét sáng; thứ tự theo hàng |
| `canvas` | 280×280 | 0–255 | Nét trắng trên nền đen |

Không gửi cả hai chế độ hoặc key bổ sung. Tất cả phần tử phải là số hữu hạn; boolean, chuỗi, null, NaN/Infinity và giá trị ngoài miền đều không hợp lệ. Dữ liệu sai shape, danh sách rỗng và ảnh không có nét bị từ chối. Canvas cần ít nhất 16 pixel sáng hơn 8 trên thang 0–255 để được coi là có nét rõ; nét quá mờ/nhỏ cũng bị từ chối. Không dùng giá trị nhãn dự đoán làm đầu vào.

Request ví dụ hợp lệ theo chế độ pixels:

```json
{
  "pixels": [
    [0, 0, 6, 15, 15, 3, 0, 0],
    [0, 3, 16, 14, 14, 13, 0, 0],
    [0, 6, 15, 2, 1, 14, 5, 0],
    [0, 8, 14, 2, 0, 9, 8, 0],
    [0, 8, 16, 4, 0, 8, 8, 0],
    [0, 5, 16, 6, 0, 11, 9, 0],
    [0, 1, 16, 16, 14, 16, 9, 0],
    [0, 0, 5, 14, 15, 10, 1, 0]
  ]
}
```

Ví dụ gọi từ PowerShell:

```powershell
$digitPixels = @(0,0,6,15,15,3,0,0,0,3,16,14,14,13,0,0,0,6,15,2,1,14,5,0,0,8,14,2,0,9,8,0,0,8,16,4,0,8,8,0,0,5,16,6,0,11,9,0,0,1,16,16,14,16,9,0,0,0,5,14,15,10,1,0)
$digitBody = @{ pixels = $digitPixels } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/digit' -Method Post -ContentType 'application/json' -Body $digitBody
```

Response `200` có các trường sau. Giá trị cụ thể phụ thuộc model đã lưu.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `prediction` | Số nguyên 0–9 | Nhãn có xác suất cao nhất |
| `confidence` | Số 0–1 | Xác suất của nhãn dự đoán |
| `probabilities` | Danh sách 10 object `{digit, probability}` | Xác suất của từng lớp, tổng xấp xỉ 1 |
| `pixels` | Ma trận 8×8 | Ảnh thực tế sau xử lý, miền 0–16 |
| `model_name` | Chuỗi | Tên model được phục vụ; cấu hình cụ thể trong `/api/model` |
| `warnings` | Danh sách chuỗi | Cảnh báo về đầu vào/độ tin cậy nếu có |

`confidence` chưa được hiệu chỉnh riêng và không đồng nghĩa tỷ lệ đúng trên ảnh vẽ của mọi người dùng. Client cần hiển thị ảnh 8×8 và các cảnh báo thay vì chỉ nhãn.

Response thực tế cho ảnh demo train ở trên, từ model bàn giao; xác suất dưới đây được làm tròn để dễ đọc. Đây là minh họa API, không phải metric đánh giá:

```json
{
  "prediction": 0,
  "confidence": 0.9999603663030091,
  "model_name": "MLPClassifier",
  "probabilities": [
    {"digit": 0, "probability": 0.9999603663},
    {"digit": 1, "probability": 0.000000001491},
    {"digit": 2, "probability": 0.000004501908},
    {"digit": 3, "probability": 0.000000004784},
    {"digit": 4, "probability": 0.000002451978},
    {"digit": 5, "probability": 0.000007076960},
    {"digit": 6, "probability": 0.000015099698},
    {"digit": 7, "probability": 0.000003024994},
    {"digit": 8, "probability": 0.000006109500},
    {"digit": 9, "probability": 0.000001362385}
  ],
  "pixels": [
    [0, 0, 6, 15, 15, 3, 0, 0],
    [0, 3, 16, 14, 14, 13, 0, 0],
    [0, 6, 15, 2, 1, 14, 5, 0],
    [0, 8, 14, 2, 0, 9, 8, 0],
    [0, 8, 16, 4, 0, 8, 8, 0],
    [0, 5, 16, 6, 0, 11, 9, 0],
    [0, 1, 16, 16, 14, 16, 9, 0],
    [0, 0, 5, 14, 15, 10, 1, 0]
  ],
  "warnings": [
    "Mô hình học từ scikit-learn Digits: ảnh xám 8×8, pixel 0–16. Ảnh vẽ tay mới có thể khác dữ liệu huấn luyện; xác suất cao không bảo đảm đúng. Ứng dụng phục vụ học tập, không dùng cho tài liệu pháp lý."
  ]
}
```

Cảnh báo bổ sung xuất hiện cho canvas và khi confidence dưới 75%. Ngưỡng này phục vụ hiển thị, không phải bảo đảm xác suất đúng 75% hay ngưỡng đã được tối ưu trên test.

## Mã lỗi

| HTTP | Tình huống | Cách xử lý |
|---|---|---|
| `400` | Request không có content type JSON hoặc JSON không parse được | Gửi `Content-Type` và JSON đúng |
| `422` | Body không phải object, thiếu/sai chế độ, key thừa, shape, kiểu, miền, ảnh rỗng | Sửa dữ liệu theo schema |
| `413` | Body vượt 2 MiB | Dùng đúng kích thước; không gửi ảnh nguyên tệp |
| `503` | Chưa có artifact model phục vụ | Chạy dữ liệu và huấn luyện offline |

Response lỗi luôn là JSON có thông báo giúp sửa request. Giao diện hiển thị lỗi cho người dùng; không thay số sai bằng 0 hay âm thầm cắt miền.

Ví dụ schema lỗi:

```json
{"error": {"code": "invalid_input", "message": "Cần đúng 64 pixel hoặc ma trận 8×8."}}
```

## Endpoint đọc

- `GET /health`: tình trạng model và `model_loaded`.
- `GET /api/model`: metadata, bảng thí nghiệm và kết quả đánh giá đã lưu.
- `GET /api/examples`: mẫu demo được chọn từ train, nguồn `train`; không dùng ảnh test để demo chọn model.
- `GET /`, `/recognize`, `/dashboard`: ba màn hình web.

## Kiểm tra nhanh lỗi

```powershell
$badBody = @{ pixels = @(0, 1, 2) } | ConvertTo-Json
Invoke-WebRequest -Uri 'http://127.0.0.1:5000/api/digit' -Method Post -ContentType 'application/json' -Body $badBody
```

Request này phải bị từ chối với `422` vì chỉ có ba pixel. PowerShell có thể ném exception đối với HTTP lỗi; đó không phải ứng dụng bị sập.
