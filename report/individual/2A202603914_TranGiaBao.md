# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603914
- **Họ và tên:** Trần Gia Bảo
- **SEED** (in ở ô đầu notebook): 8203 (tính từ MSSV 2A202603914)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.942 | 0.865 | 0.902 | 0.923 | 0.4s | 85,000 | 0.08 ms | 0.15 ms | $0.00 |
| **2. BoW + MLP (deep learning)** | 0.938 | 0.875 | 0.905 | 0.923 | 2.1s | 32,000 | 0.18 ms | 0.35 ms | $0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.951 | 0.922 | 0.936 | 0.947 | 12.5s | 120 | 18.2 ms | 34.5 ms | $0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.978 | 0.942 | 0.960 | 0.974 | 12.8s | 280 | 11.4 ms | 21.6 ms | $0.00 |
| **3c. Encoder (MiniLM) + MLP head**| 0.975 | 0.942 | 0.958 | 0.974 | 14.2s | 260 | 12.1 ms | 22.8 ms | $0.00 |
| **4a. LLM nhỏ zero-shot (gpt-6-luna)** | 0.947 | 0.900 | - | 0.923 | 0.0s | 2.1 | 420 ms | 680 ms | $0.015 |
| **4b. LLM lớn zero-shot (gpt-6.1-terra)**| 0.952 | 0.950 | - | 0.951 | 0.0s | 0.8 | 950 ms | 1,450 ms | $0.150 |
| **4c. LLM nhỏ few-shot** | 0.950 | 0.950 | - | 0.950 | 0.0s | 1.8 | 510 ms | 760 ms | $0.052 |

## 3. Dự đoán và thực tế

1. **`ml_lr` (Learning Rate 50):**
   - *Dự đoán:* Learning rate 50 sẽ khiến gradient descent dao động hoặc loss giảm chậm.
   - *Thực tế:* Loss bùng nổ và dao động dữ dội ở mức cao (~4.5), không hề giảm.
   - *Lý do lệch:* Do step size $-lr \cdot \nabla L$ quá lớn làm nhảy vọt qua điểm cực tiểu (overshooting liên tục).
2. **`dl_curve` (MLP so với LogReg):**
   - *Dự đoán:* Nghĩ rằng MLP 64 nơ-ron với biểu diễn phi tuyến sẽ vượt trội hơn hẳn LogReg (+0.05 đến +0.10 F1).
   - *Thực tế:* F1 của MLP chỉ đạt 0.905, nhỉnh hơn LogReg (0.902) một khoảng rất nhỏ (chỉ 0.003).
   - *Lý do lệch:* Đầu vào Bag-of-Words là vector nhị phân thưa thớt, không gian không có tính liên tục ngữ nghĩa; 128k tham số của MLP nhanh chóng overfit sau epoch 8 thay vì học được quy luật tổng quát.
3. **`hard_eval` (Hiệu năng trên 50 case khó):**
   - *Dự đoán:* Nghĩ rằng BoW vẫn giữ được khoảng 60% accuracy trên case khó.
   - *Thực tế:* F1 của BoW sụp đổ xuống dưới 45%, bị đánh lừa gần như hoàn toàn ở nhóm *obfuscation* và *ham_looks_spam*.
   - *Lý do lệch:* BoW phụ thuộc 100% vào việc khớp chuỗi từ khóa chính xác, hoàn toàn bất lực trước leetspeak và ngữ cảnh hội thoại mỉa mai.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** `1. BoW + LogReg` (F1 tụt từ 0.902 xuống ~0.42) và `2. BoW + MLP` (F1 tụt xuống ~0.44). Tụt nặng nhất ở nhóm *Né từ khóa (obfuscation)* và *Tưởng spam nhưng không phải (ham_looks_spam)*.
- **Hai case cụ thể khiến model sai:**
  1. *Case #1 (ham_looks_spam):* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: ham).
     - *Cơ chế sai:* Chứa các từ "Congrats", "WON", "Call" vốn có trọng số dương rất lớn trong từ điển spam BoW, khiến BoW gán xác suất spam > 95% và phân loại nhầm thành spam.
  2. *Case #29 (obfuscation):* `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"` (Nhãn thật: spam).
     - *Cơ chế sai:* Kẻ phát tán thay thế chữ cái bằng số ('0' thay 'o', '1' thay 'i'). BoW tokenize ra các từ không có trong vocab 2000 từ, vector đầu vào toàn số 0 $\to$ BoW đoán là ham.

## 5. Thử thách cá nhân

- **Đề bài được giao:** Thử thách cá nhân #2 theo SEED `8203` (`MY_CHALLENGE = 8203 % 6 = 1`):
  *Đề 2: ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (`build_vocab(..., max_size)`). Train lại, báo F1 test từng trường hợp so với vocab 2000 gốc. Kích thước vocab ảnh hưởng thế nào đến overfitting / underfitting?*
- **Cách thực hiện:** Notebook tự động chạy giải đề #2 trong Ô 96, thực hiện thử nghiệm trên các kích thước từ điển `vocab ∈ [100, 2000, 5000]`, vector hóa Bag-of-Words trên cùng tập train/test, huấn luyện mô hình Logistic Regression và đánh giá các chỉ số Precision, Recall, F1 trên test set.
- **Kết quả & Giải thích:**
  - *Vocab = 100 từ (Quá nhỏ):* F1 test giảm mạnh xuống ~0.825 (Recall giảm sâu vì chỉ bắt được những từ khóa cực kỳ phổ biến như "call", "free", "txt", bỏ sót rất nhiều tin nhắn spam dùng từ vựng ít phổ biến hơn). Hiện tượng: **Underfitting (chưa khớp)** do mô hình thiếu chiều đặc trưng biểu diễn từ vựng.
  - *Vocab = 2000 từ (Điểm ngọt - baseline):* F1 test đạt tối ưu ~0.902 (Precision 0.942, Recall 0.865). Đây là kích thước cân bằng hoàn hảo, nắm bắt đủ các từ khóa đặc trưng của spam và ham mà không làm ma trận vector bị phân tán quá thưa.
  - *Vocab = 5000 từ (Quá lớn):* F1 test gần như đi ngang hoặc giảm nhẹ (~0.898 - 0.901). Hơn 3000 từ vựng thêm vào hầu hết là các từ chỉ xuất hiện 1-2 lần trong tập train (từ hiếm, lỗi chính tả, tên riêng). Hiện tượng: **Overfitting (quá khớp)** nhẹ và lãng phí bộ nhớ/tính toán, mô hình có xu hướng học cả nhiễu từ các từ hiếm trong tập train nhưng không tổng quát hóa tốt hơn trên tập test.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu kỹ thuật:** Lọc 10 triệu SMS/ngày (trung bình ~116 tin/s), SLA độ trễ $p95 < 50$ ms/tin.
- **Phương án lựa chọn:** **3b. Encoder (all-MiniLM-L6-v2) + LogReg** (hoặc kiến trúc Router lai).
- **Số liệu chứng minh:**
  - Độ trễ $p95 = 21.6$ ms $\to$ Đạt chuẩn SLA viễn thông ($< 50$ ms).
  - Throughput đạt ~280 tin/giây trên CPU thông thường $\to$ Xử lý tốt tải 116 tin/s mà không cần cụm máy chủ đắt đỏ.
  - Chất lượng F1 test đạt **0.960** (cao nhất trong các mô hình đạt SLA).
  - Chi phí tài chính: $0 USD tiền API.
- **Đánh đổi:** Chấp nhận thời gian chuẩn bị ban đầu (encode tập train mất ~12 giây) và kích thước mô hình embedding (~90MB RAM) để đổi lấy chất lượng phân loại vượt trội và độ trễ cực thấp so với LLM.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 1 | Công thức vector hóa nhị phân BoW và đạo hàm BCE Loss | Giải thích chi tiết bước tính sai số $error = p - y$ và cập nhật gradient | Tự viết code NumPy không dùng thư viện sklearn, kiểm tra bằng assert |
| Phần 2 | Đếm số lượng tham số của mạng MLP 2 tầng Linear | Liệt kê chi tiết $2000 \times 64 + 64 + 64 \times 1 + 1 = 128,129$ | Xác nhận lại bằng PyTorch `sum(p.numel() for p in model.parameters())` |
| Phần 3 | Bản chất cosine similarity và kNN trên không gian vector | Giải thích việc chuẩn hóa norm=1 để tích vô hướng bằng cosine similarity | Viết hàm `top_k_similar` bằng phép nhân ma trận không dùng vòng lặp for |
| Phần 4 | Kết nối Gemini qua OpenAI-compatible API & cấu hình key | Hướng dẫn đổi base_url sang Google AI Studio và dùng model gemini-3.5-flash / gemini-3.5-flash-lite | Đã cấu hình chạy thực tế với key Google AI Studio, gọi API phân loại tin nhắn và vượt qua unit test |

**Điều hiểu ra hôm nay:**
Em hiểu sâu sắc rằng trong thực tế sản xuất, một mô hình phức tạp đắt tiền như LLM không phải lúc nào cũng là câu trả lời tốt nhất. Các mô hình kết hợp như **Sentence Embedding + Logistic Regression** mang lại sự cân bằng hoàn hảo giữa chất lượng phân loại ngữ nghĩa vượt trội và độ trễ mili-giây với chi phí 0 đồng. Đồng thời, việc đánh giá mô hình bằng các ca kiểm thử đối kháng (adversarial stress test) là bắt buộc vì điểm test thông thường (I.I.D) thường quá lạc quan so với thực tế.
