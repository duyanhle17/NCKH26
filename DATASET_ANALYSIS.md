# Phân tích Dataset - Chuẩn bị cho hệ thống RAG

## 1. Tổng quan

| Chỉ số | Giá trị |
|--------|---------|
| **Tổng số văn bản** | 11.280 văn bản |
| **Tổng số file** | 22.560 file (mỗi văn bản gồm 1 file `.json` + 1 file `.txt`) |
| **Số loại văn bản** | 20 thể loại |
| **Dung lượng TXT** | 134.2 MB |
| **Dung lượng JSON** | 456.7 MB |
| **Tổng dung lượng** | ~591 MB |
| **Lĩnh vực** | Thuế - Phí - Lệ phí (`Thue-Phi-Le-Phi`) |
| **Khoảng thời gian** | 1946 – 2023 |
| **Nguồn** | [Thư viện Pháp luật](https://thuvienphapluat.vn) |

---

## 2. Phân bố theo loại văn bản

| STT | Loại văn bản | Số lượng | Tỉ lệ |
|-----|-------------|----------|--------|
| 1 | Quyết định | 6.950 | 61.6% |
| 2 | Nghị quyết | 2.761 | 24.5% |
| 3 | Thông tư | 798 | 7.1% |
| 4 | Chỉ thị | 202 | 1.8% |
| 5 | Nghị định | 117 | 1.0% |
| 6 | Thông báo | 87 | 0.8% |
| 7 | Thông tư liên tịch | 75 | 0.7% |
| 8 | Điều ước quốc tế | 57 | 0.5% |
| 9 | Văn bản hợp nhất | 48 | 0.4% |
| 10 | Sắc lệnh | 47 | 0.4% |
| 11 | Lệnh | 28 | 0.2% |
| 12 | Luật | 23 | 0.2% |
| 13 | Hướng dẫn | 21 | 0.2% |
| 14 | Kế hoạch | 21 | 0.2% |
| 15 | Công điện | 11 | 0.1% |
| 16 | Pháp lệnh | 10 | 0.1% |
| 17 | Quy chế | 12 | 0.1% |
| 18 | Văn bản khác | 9 | 0.1% |
| 19 | Quy định | 2 | <0.1% |
| 20 | Thông tri | 1 | <0.1% |

> **Nhận xét:** Dataset bị **lệch phân bố** (imbalanced) – Quyết định và Nghị quyết chiếm **86.1%** tổng số văn bản.

---

## 3. Phân bố theo năm ban hành (Top 10)

| Năm | Số lượng |
|-----|----------|
| 2007 | 1.322 |
| 2009 | 1.087 |
| 2011 | 996 |
| 2012 | 970 |
| 2008 | 966 |
| 2010 | 897 |
| 2006 | 593 |
| 2017 | 502 |
| 2004 | 392 |
| 2013 | 347 |

> **Nhận xét:** Phần lớn văn bản tập trung giai đoạn **2006–2012**, phù hợp với thời kỳ cải cách thuế phí tại Việt Nam.

---

## 4. Cơ quan ban hành (Top 10)

| Cơ quan | Số lượng |
|---------|----------|
| Bộ Tài chính | 1.211 |
| Thành phố Hà Nội | 554 |
| Tỉnh Khánh Hòa | 327 |
| Tỉnh Lâm Đồng | 282 |
| Thành phố Hồ Chí Minh | 281 |
| Tỉnh Lào Cai | 261 |
| Tỉnh Thừa Thiên Huế | 259 |
| Tỉnh Đồng Tháp | 236 |
| Tỉnh An Giang | 235 |
| Tỉnh Bình Thuận | 228 |

> **Nhận xét:** Bộ Tài chính là nguồn lớn nhất. Phần còn lại phân bố đều từ UBND các tỉnh/thành.

---

## 5. Tình trạng hiệu lực

| Tình trạng | Số lượng | Tỉ lệ |
|------------|----------|--------|
| Đã biết (chưa rõ) | 10.515 | 93.2% |
| Hết hiệu lực | 486 | 4.3% |
| Còn hiệu lực | 182 | 1.6% |
| Không xác định | 44 | 0.4% |
| Hết hiệu lực một phần | 40 | 0.4% |
| Không có thông tin | 11 | 0.1% |
| Tạm ngưng hiệu lực | 2 | <0.1% |

---

## 6. Cấu trúc dữ liệu

### 6.1. File TXT – Văn bản thuần (Plain text)

- Nội dung văn bản pháp luật dạng **text thuần**, không có markup
- Kích thước: **749 bytes → 195 KB**, trung bình **~18 KB**
- Phù hợp để dùng trực tiếp cho **chunking** và **embedding**

### 6.2. File JSON – Dữ liệu có cấu trúc

Mỗi file JSON gồm 2 phần chính:

#### a) `document_info` – Metadata

| Trường | Mô tả | Ví dụ |
|--------|-------|-------|
| `title` | Tiêu đề văn bản | "Luật Thuế thu nhập cá nhân 2007" |
| `so_hieu` | Số hiệu văn bản | "04/2007/QH12" |
| `loai_van_ban` | Loại văn bản | "Luật", "Nghị định", "Quyết định"... |
| `category` | Lĩnh vực | "Thue-Phi-Le-Phi" |
| `link` | URL gốc trên thuvienphapluat.vn | https://thuvienphapluat.vn/... |
| `ngay_ban_hanh` | Ngày ban hành | "21/11/2007" |
| `noi_ban_hanh` | Cơ quan ban hành | "Quốc hội", "Bộ Tài chính"... |
| `tinh_trang` | Tình trạng hiệu lực | "Còn hiệu lực", "Hết hiệu lực"... |
| `ngay_hieu_luc`* | Ngày có hiệu lực | "15/02/1999" |
| `su_kien_phap_ly`* | Lịch sử sự kiện pháp lý | Danh sách các sự kiện (ban hành, hiệu lực, hết hiệu lực, bãi bỏ) |
| `do_khop_vbpl`* | Mức độ khớp với VBPL | "exact" |
| `vbpl_item_id`* | ID trên hệ thống VBPL | 7245 |
| `enriched_at`* | Thời điểm bổ sung dữ liệu | "2026-02-08T17:24:57" |

> (*) Các trường có dấu `*` chỉ xuất hiện ở **~6.4%** văn bản (719/11.280) – là phần được **enriched** thêm.

#### b) `parsed_result.structure` – Cây cấu trúc văn bản

Văn bản được parse thành **cây phân cấp (hierarchical tree)** với các loại node:

| Node type | Ý nghĩa | Cấp |
|-----------|---------|-----|
| `document` | Gốc tài liệu | 0 |
| `part` | Phần (Phần I, Phần II...) | 1 |
| `chapter` | Chương | 2 |
| `section` | Mục | 3 |
| `article` | Điều | 3-4 |
| `clause` | Khoản | 4-5 |
| `point` | Điểm (a, b, c...) | 5-6 |
| `item` | Tiết / Mục nhỏ | 6 |

**Thống kê cấu trúc (mẫu 182 văn bản):**
- Số node trung bình: **41 node/văn bản** (min: 1, max: 908)
- Độ sâu tối đa trung bình: **2 cấp** (min: 0, max: 6)

Mỗi node có các thuộc tính:
- `type`: Loại node
- `title`: Tiêu đề (vd: "Điều 1. Phạm vi điều chỉnh")
- `html_id`: ID tham chiếu (vd: "dieu_1", "chuong_1")
- `content`: Nội dung text của node
- `children`: Danh sách các node con

---

## 7. Đặc điểm quan trọng cho hệ thống RAG

### ✅ Điểm mạnh

1. **Dữ liệu có cấu trúc phân cấp rõ ràng** – Cây cấu trúc `document → chapter → article → clause → point` cho phép chunking theo ngữ nghĩa pháp lý thay vì cắt cơ học
2. **Metadata phong phú** – Có đầy đủ thông tin `số hiệu`, `ngày ban hành`, `cơ quan`, `tình trạng hiệu lực` giúp lọc và ranking kết quả
3. **Dual format (JSON + TXT)** – JSON phục vụ chunking thông minh, TXT phục vụ full-text search
4. **Quy mô lớn** – 11.280 văn bản, ~591 MB, đủ để xây dựng hệ thống RAG có chiều sâu
5. **Một lĩnh vực tập trung** – Thuế-Phí-Lệ phí → domain-specific, giúp embedding và retrieval chính xác hơn
6. **Tham chiếu chéo** – Nhiều văn bản tham chiếu lẫn nhau qua `su_kien_phap_ly`, hỗ trợ xây dựng Knowledge Graph

### ⚠️ Thách thức

1. **Phân bố lệch** – Quyết định + Nghị quyết chiếm 86%, có thể ảnh hưởng đến retrieval diversity
2. **Tình trạng hiệu lực không rõ** – 93% văn bản có trạng thái "Đã biết" (ambiguous), khó lọc văn bản còn/hết hiệu lực
3. **Enrichment chưa đầy đủ** – Chỉ 6.4% văn bản có thông tin `su_kien_phap_ly` và `ngay_hieu_luc`
4. **Kích thước văn bản biến thiên lớn** – Từ <1KB đến ~195KB, cần chiến lược chunking linh hoạt
5. **Nội dung lặp trong JSON** – Một số node có content bị duplicate (xuất hiện 2 lần), cần xử lý deduplicate
6. **Ngôn ngữ tiếng Việt** – Cần tokenizer/embedding model hỗ trợ tốt tiếng Việt

### 🔧 Khuyến nghị cho pipeline RAG

| Bước | Khuyến nghị |
|------|-------------|
| **Chunking** | Sử dụng cấu trúc JSON (Điều/Khoản/Điểm) làm đơn vị chunk thay vì cắt theo token cố định |
| **Metadata filtering** | Gắn metadata (`loai_van_ban`, `so_hieu`, `ngay_ban_hanh`, `noi_ban_hanh`, `tinh_trang`) vào mỗi chunk để hỗ trợ hybrid search |
| **Embedding model** | Sử dụng model hỗ trợ tiếng Việt tốt (vd: `bkai-foundation-models/vietnamese-bi-encoder`, `keepitreal/vietnamese-sbert`) |
| **Knowledge Graph** | Tận dụng `su_kien_phap_ly` và tham chiếu chéo giữa các văn bản để xây dựng graph liên kết |
| **Deduplication** | Xử lý content lặp trong JSON trước khi chunk |
| **Hierarchical retrieval** | Lưu cả context cha (Chương → Điều) bên cạnh chunk để cải thiện ngữ cảnh trả về |

---

## 8. Ví dụ cấu trúc JSON

```json
{
  "document_info": {
    "title": "Luật Thuế thu nhập cá nhân 2007",
    "so_hieu": "04/2007/QH12",
    "loai_van_ban": "Luật",
    "category": "Thue-Phi-Le-Phi",
    "link": "https://thuvienphapluat.vn/...",
    "ngay_ban_hanh": "21/11/2007",
    "noi_ban_hanh": "Quốc hội",
    "tinh_trang": "Đã biết"
  },
  "parsed_result": {
    "structure": {
      "type": "document",
      "title": "Luật Thuế thu nhập cá nhân 2007",
      "children": [
        {
          "type": "chapter",
          "title": "Chương 1:",
          "html_id": "chuong_1",
          "content": "NHỮNG QUY ĐỊNH CHUNG",
          "children": [
            {
              "type": "article",
              "title": "Điều 1. Phạm vi điều chỉnh",
              "html_id": "dieu_1",
              "content": "Luật này quy định về đối tượng nộp thuế..."
            },
            {
              "type": "article",
              "title": "Điều 2. Đối tượng nộp thuế",
              "html_id": "dieu_2",
              "children": [
                {
                  "type": "clause",
                  "title": "1",
                  "content": "Đối tượng nộp thuế thu nhập cá nhân là..."
                },
                {
                  "type": "clause",
                  "title": "2",
                  "content": "Cá nhân cư trú là người đáp ứng...",
                  "children": [
                    { "type": "point", "title": "a) Có mặt tại Việt Nam..." },
                    { "type": "point", "title": "b) Có nơi ở thường xuyên..." }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  }
}
```
