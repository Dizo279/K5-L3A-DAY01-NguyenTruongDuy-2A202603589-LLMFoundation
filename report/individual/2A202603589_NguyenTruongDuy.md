# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603589
- **Họ và tên:** Nguyễn Trường Duy
- **SEED** (in ở ô đầu notebook): 5000

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| | precision | recall | F1 (test) | F1 (SUB 40)|
|-|-----------|--------|-----------|-------------|
|1. BoW + LogReg (tự code) | 1.000 | 0.809 |  0.894  | 0.788 |
|2. BoW + MLP (deep learning) | 0.975 | 0.888 |  0.929  | 0.919 |
|3a. Encoder (MiniLM) + kNN | 0.885 | 0.955 |  0.919  | 0.976 |
|3b. Encoder (MiniLM) + LogReg | 0.977 | 0.944 |  0.960  | 1.000 |
|3c. Encoder (MiniLM) + MLP head | 0.977 | 0.955 |  0.966  | 1.000 |
|4a. LLM nhỏ zero-shot | 0.870 | 1.000 |  0.930  | 0.930 |
|4b. LLM lớn zero-shot | 0.870 | 1.000 |  0.930  | 0.930 |
|4c. LLM nhỏ few-shot | 0.870 | 1.000 |  0.930  | 0.930 |

| | train (s) | tin/s (cả lô) | p50 ms/tin |
|-|-----------|------------|-------------|
|1. BoW + LogReg (tự code) | 8.749 | 10876.532 | 0.046 |
|2. BoW + MLP (deep learning) | 10.233 | 12372.169 | 0.236 |
|3a. Encoder (MiniLM) + kNN | 2.658 | 44.004 | 49.360 |
|3b. Encoder (MiniLM) + LogReg | 3.860 | 1196.740 | 11.505 |
|3c. Encoder (MiniLM) + MLP head | 3.956 | 1987.269 | 6.602 |
|4a. LLM nhỏ zero-shot | 0.000 | 1.071 | 325.150 |
|4b. LLM lớn zero-shot | 0.000 | 1.291 | 296.950 |
|4c. LLM nhỏ few-shot | 0.000 | 0.639 | 2246.787 |

| | p95 ms/tin | USD / 1k tin |
|-|------------|--------------|
|1. BoW + LogReg (tự code) | 0.110 | 0.0 |
|2. BoW + MLP (deep learning) | 2.298 | 0.0 |
|3a. Encoder (MiniLM) + kNN | 66.664 | 0.0 |
|3b. Encoder (MiniLM) + LogReg | 17.673 | 0.0 |
|3c. Encoder (MiniLM) + MLP head | 7.344 | 0.0 |
|4a. LLM nhỏ zero-shot | 2412.345 | 0.0 |
|4b. LLM lớn zero-shot | 2307.225 | 0.0 |
|4c. LLM nhỏ few-shot | 3259.186 | 0.0 |

## 3. Dự đoán và thực tế

1. Learning rate quá lớn (lr_50)
- Dự đoán: Loss tăng và dao động mạnh.
- Kết quả: Loss không hội tụ ổn định.
- Lý do: Learning rate quá lớn khiến mô hình cập nhật trọng số vượt quá điểm tối ưu.
2. Few-shot LLM
- Dự đoán: F1 sẽ tăng so với Zero-shot.
- Kết quả: F1 không thay đổi, vẫn là 0.930.
- Lý do: Các ví dụ bổ sung không cung cấp thêm thông tin hữu ích.
3. MLP 512 neuron
- Dự đoán: F1 sẽ cao hơn MLP 4 neuron.
- Kết quả: Cần đối chiếu kết quả thực nghiệm của hai mô hình.
- Lý do: Nhiều neuron hơn không đảm bảo tổng quát hóa tốt hơn, có thể gây overfitting.

## 4. Phân tích 50 case khó

BoW + Logistic Regression tụt nhiều nhất, F1 giảm từ 0.788 xuống 0.333. Khó nhất ở nhóm natural_spam và obfuscation.

Hai case cụ thể:
1. C0ngr4tulat10ns: BoW khó nhận diện từ bị thay chữ bằng số vì không khớp từ khóa trong vocabulary.
2. Tin spam có nội dung thân thiện, không chứa từ khóa như FREE, WIN, PRIZE: BoW dễ phân loại nhầm thành Ham vì chỉ dựa vào sự xuất hiện của từ.
Nguyên nhân: BoW không hiểu ngữ cảnh và ngữ nghĩa, chủ yếu dựa vào từ khóa đã học nên dễ sai với tin nhắn mới hoặc bị biến đổi.

## 5. Thử thách cá nhân

## 6. Phương án cho sản phẩm thật

Em chọn Embedding + kNN vì F1 test đạt 0.976, cao hơn BoW và có chất lượng tốt trên 50 case khó (F1 = 0.759).
Mô hình này phù hợp hơn LLM về chi phí và tiềm năng độ trễ. Tuy nhiên, cần kiểm tra p95 thực tế để xác nhận đạt dưới 50 ms/tin.
Đánh đổi: Chấp nhận F1 thấp hơn LLM có phòng thủ (0.935 trên tập khó) để đổi lấy chi phí thấp và tốc độ xử lý tốt hơn. Đồng thời cần tối ưu tìm kiếm láng giềng khi dữ liệu lớn.
## 7. AI log và điều đã học

1. Hỏi cách triển khai và so sánh các mô hình ML, DL, Embedding và LLM. Nhận code và giải thích, sau đó điều chỉnh tên biến và tham số cho phù hợp notebook.
2. Hỏi cách đánh giá mô hình trên dữ liệu khó. Nhận phân tích về F1 và các lỗi phân loại, sau đó đối chiếu với kết quả thực nghiệm.
3. Hỏi cách chọn mô hình cho hệ thống lọc SMS. Nhận đề xuất Embedding + kNN và Router kết hợp LLM.
Điều đã học:

Em hiểu rằng F1 cao trên tập test thông thường chưa đảm bảo mô hình hoạt động tốt với dữ liệu thực tế. Việc lựa chọn mô hình cần cân bằng giữa độ chính xác, độ trễ và chi phí, đồng thời phải kiểm thử trên các tình huống khó.