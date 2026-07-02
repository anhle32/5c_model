# 🏦 Ứng dụng Dự báo Rủi ro Tín dụng theo Khung 5C

Ứng dụng web Streamlit chuyển thể từ notebook huấn luyện mô hình **Logistic Regression** dự báo khả năng vỡ nợ (PD) của khách hàng, dựa trên **24 tiêu chí Likert (1–5)** theo khung 5C:

| Nhóm 5C | Biến | Số biến |
|---|---|---|
| Tư cách (Character) | TC1–TC5 | 5 |
| Năng lực (Capacity) | NL1–NL4 | 4 |
| Điều kiện (Conditions) | DK1–DK5 | 5 |
| Vốn (Capital) | V1–V6 | 6 |
| Tài sản đảm bảo (Collateral) | TS1–TS4 | 4 |

Biến mục tiêu: `PD` (0 = không rủi ro, 1 = rủi ro vỡ nợ).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc file dữ liệu đầu vào

- Định dạng: **CSV** (hỗ trợ BOM UTF-8, ví dụ xuất từ Google Forms).
- Bắt buộc có 25 cột: 24 biến đầu vào `TC1…TC5, NL1…NL4, DK1…DK5, V1…V6, TS1…TS4` (giá trị nguyên 1–5) và cột mục tiêu `PD` (0/1).
- Các cột khác (ví dụ `Dấu thời gian`, `NN`) được phép có mặt nhưng **không** đưa vào mô hình — đúng như notebook gốc.
- Ở chế độ dự báo hàng loạt (tab Sử dụng mô hình), tệp tải lên chỉ cần đủ **24 cột biến đầu vào** (không cần cột `PD`).

## Mô tả các tab

1. **📊 Tổng quan dữ liệu** — số dòng/cột/dung lượng tệp, xem dữ liệu thô (cuộn gọn), thống kê mô tả chỉ cho 24 biến X và biến PD.
2. **📈 Trực quan hóa dữ liệu** — biểu đồ cột phân phối (biến Likert rời rạc và biến mục tiêu PD); do có hơn 4 biến nên có `multiselect` để chọn biến, mặc định 4 biến ưu tiên (PD đứng đầu). Bố cục lưới 2 cột, chiều cao cố định.
3. **🧪 Kết quả huấn luyện & kiểm định** — Accuracy (tương ứng `model.score` trong notebook), Precision, Recall, F1, ROC-AUC; ma trận nhầm lẫn (tái hiện heatmap của notebook bằng Plotly); đường cong ROC; classification report; bảng chấm điểm tập test.
4. **🔮 Sử dụng mô hình** — hai chế độ:
   - *Nhập trực tiếp*: form 24 ô nhập theo nhóm 5C (mặc định = trung vị, min/max theo dữ liệu) → dự báo PD + xác suất rủi ro (tái hiện `predict` + `predict_proba` của notebook).
   - *Tải file theo cấu trúc X_test*: kiểm tra schema (báo rõ cột thiếu), dự báo hàng loạt, xuất CSV kết quả (utf-8-sig).

## Cách vận hành

- Mọi cấu hình đặt ở **sidebar**: tải tệp, `test_size` (mặc định **0.1**), `random_state` (mặc định **23**), tham số nâng cao Logistic Regression (`C`, `max_iter`, `solver` — mặc định theo notebook/sklearn).
- Huấn luyện **chỉ chạy một lần** khi bấm nút 🚀; mô hình đã fit, bộ tiền xử lý và bảng kết quả được lưu vào `st.session_state` nên chuyển tab không train lại.

## Ghi chú kỹ thuật

- Notebook gốc **không dùng scaler/encoder** (dữ liệu Likert đưa thẳng vào mô hình), app tái hiện đúng như vậy (`preprocessor = None`).
- Notebook dùng siêu tham số mặc định của `LogisticRegression()`; sidebar hiển thị các giá trị mặc định này và cho phép tinh chỉnh.
- Notebook không nêu tên đề tài; tiêu đề app được suy từ tên biến (khung 5C) và bối cảnh dự báo PD — đây là giả định hợp lý được ghi chú tại đây.
- Khuyến nghị **Streamlit ≥ 1.38** (đã kiểm thử tốt với `st.container(height=...)`, `st.toast`); các bản mới hơn (≥1.55) hỗ trợ thêm `st.container(horizontal=True)`, `st.space`, dynamic container nếu muốn mở rộng.
