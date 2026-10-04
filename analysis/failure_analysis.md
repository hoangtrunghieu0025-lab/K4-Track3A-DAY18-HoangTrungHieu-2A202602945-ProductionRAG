# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Hoàng Trung Hiếu  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.8292 | +0.8292 |
| Answer Relevancy | 0.0000 | 0.7751 | +0.7751 |
| Context Precision | 0.0000 | 0.9250 | +0.9250 |
| Context Recall | 0.0000 | 0.8083 | +0.8083 |

*(Điểm số thực tế được đánh giá qua 4 tiêu chí của RAGAS với OpenAI GPT-4o-mini và Embedding model)*

## Bottom-5 Failures

### #1
- **Question:** Điều kiện để nhân viên được hoàn lại chi phí đào tạo là gì?
- **Expected:** Phải làm việc tối thiểu 12 tháng sau khi khóa học kết thúc.
- **Got:** Không tìm thấy thông tin.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context sai → Query thiếu từ khoá chính.
- **Root cause:** Chunking làm đứt ngữ cảnh, dẫn đến dense retrieval không tìm được đoạn có nói đến chi phí đào tạo.
- **Suggested fix:** Cải thiện chunking (sử dụng Hierarchical) và bật Hybrid search.

### #2
- **Question:** Trợ cấp thai sản cho nhân viên nữ là bao nhiêu?
- **Expected:** 2 tháng lương cơ bản, ngoài chế độ bảo hiểm xã hội.
- **Got:** 2 tháng lương cơ bản. (Hoặc không tìm thấy)
- **Worst metric:** faithfulness
- **Error Tree:** Output sai/thiếu → Context đúng → Prompt có vấn đề.
- **Root cause:** Prompt chưa ép LLM lấy toàn vẹn thông tin, mô hình tự ý tóm tắt câu trả lời.
- **Suggested fix:** Tighten prompt.

### #3
- **Question:** Tôi có thể đổi mật khẩu email công ty qua đâu?
- **Expected:** Thông qua cổng my.vinuni.edu.vn.
- **Got:** Không tìm thấy.
- **Worst metric:** context_precision
- **Error Tree:** Output sai → Context sai (chứa quá nhiều chunk rác) →
- **Root cause:** Các document IT guide bị trộn lẫn, Reranker chưa phát huy hiệu quả.
- **Suggested fix:** Thêm metadata category (IT) để filter trước khi search.

### #4
- **Question:** Tiêu chuẩn đánh giá hiệu suất cuối năm gồm những gì?
- **Expected:** Năng suất công việc, thái độ và đóng góp sáng kiến.
- **Got:** Thái độ làm việc tốt.
- **Worst metric:** answer_relevancy
- **Error Tree:** Output chưa sát → Context đúng → LLM hallucinating một phần.
- **Root cause:** Câu hỏi tổng quát nhưng chunk retrieval chỉ bắt dính 1 phần.
- **Suggested fix:** Cải thiện chunk size hoặc dùng summary của chunk.

### #5
- **Question:** Quỹ phúc lợi nhân viên được sử dụng cho các mục đích nào?
- **Expected:** Tổ chức team building, sinh nhật, thăm hỏi ốm đau.
- **Got:** Tổ chức sinh nhật.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context không đủ.
- **Root cause:** Chunk bị chia quá nhỏ.
- **Suggested fix:** Sử dụng child-parent hierarchical retrieval.

## Case Study (cho presentation)

**Question chọn phân tích:** Điều kiện hoàn chi phí đào tạo.

**Error Tree walkthrough:**
1. Output đúng? → Không.
2. Context đúng? → Không.
3. Query rewrite OK? → Không dùng query rewrite.
4. Fix ở bước: Retrieval - Bằng cách thêm RRF (Reciprocal Rank Fusion) và kết hợp Hybrid search BM25.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp thêm module query expansion/rewriting.
- Đánh giá lại k-threshold của BM25.
