# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Hoàng Trung Hiếu
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold cosine similarity tạo chunks theo ngữ nghĩa thay vì ngẫu nhiên cắt ngang câu, giúp context mạch lạc hơn. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Sử dụng RRF kết hợp điểm xếp hạng lexical (từ khóa chính xác từ underthesea segment) và semantic (embedding), cải thiện recall đáng kể. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Cross encoder đánh giá sự phù hợp của query và từng document cụ thể, độ chính xác top 3 kết quả cải thiện vượt trội. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá qua bộ 4 chỉ số (Faithfulness, Relevancy, Precision, Recall) phát hiện các vấn đề retrieval hay hallucination rõ ràng. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Bổ sung câu ngữ cảnh mô tả trước chunk giúp bù đắp thông tin bị mất mát trong quá trình chunking. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - Thiếu thư viện underthesea dẫn đến lỗi ModuleNotFoundError trong lúc chạy tests M2. Lỗi `numpy` build từ source khi cài RAGAS.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Underthesea không có sẵn trong môi trường, yêu cầu cài đặt tay. Ragas 0.1.22 xung đột numpy trên python 3.13, giải quyết bằng cách cài riêng phần backend search trước.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu cách hybrid search xử lý token tiếng việt, phải thêm _replace('_', ' ')_ vào token output của Underthesea để BM25Okapi hoạt động tốt.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production-Ready RAG cho Hệ thống Tra cứu Nội bộ

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng raw text splitting bằng langchain text_splitter, dense retrieval với Qdrant và OpenAI model sinh text.
- **Vấn đề / Bottlenecks đang gặp:** Context chunking bị mất ý do cắt ngang văn bản, dense retrieval bị trượt các thuật ngữ chuyên môn.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang Hierarchical (Parent-Child) chunking để tăng độ chính xác trong truy xuất nhưng vẫn cấp đủ context cho LLM.
2. **Search retrieval:** Áp dụng Hybrid Search (BM25 + Dense) kết hợp underthesea tokenize và RRF để tận dụng ưu điểm bắt keywords chuyên ngành.
3. **Reranking:** Dùng bge-reranker-v2-m3 qua mô hình CrossEncoder nhằm lọc nhiễu top K kết quả thô, giữ lại 3 văn bản chất lượng nhất.
4. **Evaluation:** Thiết lập RAGAS với test dataset tĩnh để theo dõi pipeline regressions.
5. **Enrichment:** Gọi 1 lần API LLM cho Chunk enrichment (Context prepend + HyQA + auto metadata) nhằm tiết kiệm chi phí mà tăng chất lượng chunk embed.

#### 3. Timeline triển khai
- **Tuần 1:** Cải tổ hệ thống Indexing (M1, M2, M5) và test.
- **Tuần 2:** Áp dụng Reranking và thiết lập RAGAS Eval pipeline tự động.
