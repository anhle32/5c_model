# =============================================================
# ỨNG DỤNG DỰ BÁO RỦI RO TÍN DỤNG THEO KHUNG 5C
# Mô hình: Logistic Regression (tái hiện từ notebook gốc)
# =============================================================

import streamlit as st

# ---- 1) set_page_config: LỆNH STREAMLIT ĐẦU TIÊN ----
st.set_page_config(
    layout="wide",
    page_title="Dự báo rủi ro tín dụng 5C",
    page_icon="🏦",
)

# ---- 2) Import & các hàm cache dùng chung ----
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# Tập biến đầu vào X và biến mục tiêu y — trích chính xác từ notebook
FEATURES = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET = "PD"

# Nhóm biến theo khung 5C (phục vụ trình bày form nhập liệu)
GROUPS_5C = {
    "Tư cách (Character - TC)": ["TC1", "TC2", "TC3", "TC4", "TC5"],
    "Năng lực (Capacity - NL)": ["NL1", "NL2", "NL3", "NL4"],
    "Điều kiện (Conditions - DK)": ["DK1", "DK2", "DK3", "DK4", "DK5"],
    "Vốn (Capital - V)": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "Tài sản đảm bảo (Collateral - TS)": ["TS1", "TS2", "TS3", "TS4"],
}


@st.cache_data(show_spinner="Đang nạp dữ liệu...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Hàm nạp dữ liệu dùng chung (nhận bytes để hashable).

    Tái hiện đúng bước đọc dữ liệu của notebook: pandas.read_csv.
    Notebook không tạo biến phái sinh nên chỉ đọc và trả về DataFrame.
    """
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    return df


def validate_columns(df: pd.DataFrame, required: list) -> list:
    """Trả về danh sách cột bị thiếu so với yêu cầu."""
    return [c for c in required if c not in df.columns]


# =============================================================
# ---- 3) SIDEBAR — VÙNG CẤU HÌNH ----
# =============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu (.csv)",
        type=["csv"],
        help="Tệp CSV khảo sát 5C, chứa 24 biến Likert (TC, NL, DK, V, TS) và cột mục tiêu PD (0 = không rủi ro, 1 = rủi ro).",
    )

    # Notebook chỉ dùng MỘT mô hình (LogisticRegression) → không có selectbox chọn mô hình

    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm định (test_size)",
        min_value=0.05, max_value=0.5, value=0.1, step=0.05,
        help="Tỷ lệ dữ liệu dành cho tập test. Notebook gốc dùng 0.1 (10%).",
    )
    random_state = st.number_input(
        "Random state",
        min_value=0, max_value=9999, value=23, step=1,
        help="Hạt giống ngẫu nhiên khi chia train/test. Notebook gốc dùng 23 để tái lập kết quả.",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        c_value = st.slider(
            "Hệ số điều chuẩn C",
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help="Nghịch đảo cường độ điều chuẩn. Notebook dùng mặc định C=1.0.",
        )
        max_iter = st.number_input(
            "Số vòng lặp tối đa (max_iter)",
            min_value=100, max_value=5000, value=100, step=100,
            help="Số vòng lặp tối đa của thuật toán tối ưu. Mặc định sklearn là 100.",
        )
        solver = st.selectbox(
            "Thuật toán tối ưu (solver)",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Notebook dùng mặc định lbfgs.",
        )

    st.divider()
    run_training = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# =============================================================
# ---- 4) HEADER — VÙNG ĐỊNH HƯỚNG ----
# =============================================================
st.title("🏦 Dự báo rủi ro tín dụng theo khung 5C")
st.caption(
    "Ứng dụng huấn luyện mô hình **Logistic Regression** để dự báo khả năng vỡ nợ (PD) "
    "của khách hàng dựa trên 24 tiêu chí Likert (1–5) theo khung 5C: "
    "Tư cách (TC), Năng lực (NL), Điều kiện (DK), Vốn (V), Tài sản đảm bảo (TS). "
    "Đầu vào kỳ vọng: tệp CSV khảo sát chứa các cột TC1–TC5, NL1–NL4, DK1–DK5, V1–V6, TS1–TS4 và PD."
)

# Kiểm tra trạng thái dữ liệu
if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu CSV ở thanh bên trái để bắt đầu.")
    st.stop()

try:
    df = load_data(uploaded_file.getvalue())
except Exception as e:
    st.error(f"❌ Không đọc được tệp dữ liệu. Vui lòng kiểm tra định dạng CSV. Chi tiết: {e}")
    st.stop()

if df.empty:
    st.error("❌ Tệp dữ liệu rỗng. Vui lòng kiểm tra lại nội dung tệp.")
    st.stop()

missing_cols = validate_columns(df, FEATURES + [TARGET])
if missing_cols:
    st.error(
        "❌ Tệp dữ liệu thiếu các cột bắt buộc sau: "
        + ", ".join(f"`{c}`" for c in missing_cols)
    )
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"Tóm tắt nhanh: {df.shape[0]:,} dòng × {df.shape[1]} cột — "
    f"tỷ lệ rủi ro (PD=1): {df[TARGET].mean():.1%}"
)
st.divider()

# =============================================================
# ---- 5) KHỐI HUẤN LUYỆN (chạy khi bấm nút, lưu session_state) ----
# =============================================================
if run_training:
    try:
        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(
            C=c_value, max_iter=int(max_iter), solver=solver
        )
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yproba_test = model.predict_proba(X_test)[:, 1]

        # Bảng kết quả đã chấm điểm trên tập test
        results_df = X_test.copy()
        results_df["PD_thực_tế"] = y_test.values
        results_df["PD_dự_báo"] = yhat_test
        results_df["Xác_suất_rủi_ro"] = yproba_test

        # Lưu 3 thứ vào session_state: (1) mô hình đã fit,
        # (2) bộ tiền xử lý (notebook KHÔNG dùng scaler/encoder → None),
        # (3) bảng kết quả đã chấm điểm
        st.session_state["model"] = model
        st.session_state["preprocessor"] = None  # notebook không có tiền xử lý
        st.session_state["results_df"] = results_df
        st.session_state["eval"] = {
            "y_test": y_test,
            "yhat_test": yhat_test,
            "yproba_test": yproba_test,
        }
        st.toast("✅ Huấn luyện mô hình thành công!", icon="🎉")
    except Exception as e:
        st.error(f"❌ Lỗi khi huấn luyện mô hình: {e}")

# =============================================================
# ---- 6) CÁC TAB NỘI DUNG ----
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Tổng quan dữ liệu",
        "📈 Trực quan hóa dữ liệu",
        "🧪 Kết quả huấn luyện & kiểm định",
        "🔮 Sử dụng mô hình",
    ]
)

# -------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# -------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng tệp", f"{uploaded_file.size / (1024 * 1024):.2f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=320):
        st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình")
    st.caption("Chỉ mô tả 24 biến đầu vào (X) và biến mục tiêu PD (y) — đúng tập biến notebook sử dụng.")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# -------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# -------------------------------------------------------------
with tab2:
    st.caption(
        "Biến mục tiêu PD được ưu tiên hiển thị đầu tiên. "
        "Các biến Likert rời rạc (1–5) được vẽ dạng cột theo tần suất."
    )

    # >4 biến → multiselect, mặc định 4 biến ưu tiên (PD trước)
    default_vars = [TARGET, "TC1", "NL1", "TS1"]
    selected_vars = st.multiselect(
        "Chọn biến cần trực quan hóa (tối đa nên chọn 4 để bố cục cân đối)",
        options=[TARGET] + FEATURES,
        default=default_vars,
        help="Biến mục tiêu PD nên đứng đầu. Các biến còn lại là thang Likert 1–5.",
    )

    if not selected_vars:
        st.info("Vui lòng chọn ít nhất một biến để vẽ biểu đồ.")
    else:
        CHART_HEIGHT = 350

        def make_chart(var: str):
            """Chọn loại biểu đồ theo kiểu biến thực tế."""
            if var == TARGET:
                # Mục tiêu phân loại nhị phân → bar phân phối lớp
                counts = df[var].value_counts().sort_index()
                fig = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    labels={"x": "PD (0 = không rủi ro, 1 = rủi ro)", "y": "Số quan sát"},
                    title=f"Phân phối lớp biến mục tiêu {var}",
                    color=counts.index.astype(str),
                    color_discrete_sequence=["#2E86AB", "#E4572E"],
                )
                fig.update_layout(showlegend=False, height=CHART_HEIGHT)
            else:
                # Biến Likert rời rạc (int 1–5) → bar theo value_counts
                counts = df[var].value_counts().sort_index()
                fig = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    labels={"x": f"Mức Likert của {var}", "y": "Số quan sát"},
                    title=f"Phân phối biến {var}",
                    color_discrete_sequence=["#2E86AB"],
                )
                fig.update_layout(height=CHART_HEIGHT)
            return fig

        # Lưới 2 cột, mỗi hàng 2 biểu đồ
        for i in range(0, len(selected_vars), 2):
            cols = st.columns(2)
            for j, var in enumerate(selected_vars[i : i + 2]):
                with cols[j]:
                    st.plotly_chart(make_chart(var), use_container_width=True)

# -------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# -------------------------------------------------------------
with tab3:
    if "model" not in st.session_state:
        st.info("👈 Chưa có mô hình. Vui lòng bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        ev = st.session_state["eval"]
        y_test = ev["y_test"]
        yhat_test = ev["yhat_test"]
        yproba_test = ev["yproba_test"]

        # ---- Chỉ tiêu vô hướng (phân loại có giám sát) ----
        acc = accuracy_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test, zero_division=0)
        rec = recall_score(y_test, yhat_test, zero_division=0)
        f1 = f1_score(y_test, yhat_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yproba_test)
        except ValueError:
            auc = None  # tập test chỉ có 1 lớp

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}", help="Tương ứng model.score(X_test, y_test) trong notebook.")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "N/A")

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, yhat_test)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Dự báo 0", "Dự báo 1"],
                y=["Thực tế 0", "Thực tế 1"],
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            st.subheader("Đường cong ROC")
            if auc is not None:
                fpr, tpr, _ = roc_curve(y_test, yproba_test)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             name="Ngẫu nhiên", line=dict(dash="dash")))
                fig_roc.update_layout(
                    height=400,
                    xaxis_title="Tỷ lệ dương tính giả (FPR)",
                    yaxis_title="Tỷ lệ dương tính thật (TPR)",
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Tập test chỉ có một lớp nên không tính được ROC-AUC.")

        st.subheader("Classification report")
        report = classification_report(y_test, yhat_test, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.subheader("Bảng kết quả chấm điểm trên tập test")
        with st.container(height=320):
            st.dataframe(st.session_state["results_df"], use_container_width=True)

# -------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# -------------------------------------------------------------
with tab4:
    if "model" not in st.session_state:
        st.info("👈 Chưa có mô hình. Vui lòng bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        model = st.session_state["model"]
        # preprocessor = st.session_state["preprocessor"]  # None: notebook không tiền xử lý

        mode = st.radio(
            "Chọn chế độ dự báo",
            ["✍️ Nhập trực tiếp", "📂 Tải file theo cấu trúc X_test"],
            horizontal=True,
        )

        # ---------- CHẾ ĐỘ 1: NHẬP TRỰC TIẾP ----------
        if mode == "✍️ Nhập trực tiếp":
            st.caption(
                "Nhập mức đánh giá Likert (1–5) cho 24 tiêu chí 5C của khách hàng. "
                "Giá trị mặc định là trung vị của dữ liệu đã tải."
            )
            with st.form("form_predict"):
                inputs = {}
                for group_name, group_vars in GROUPS_5C.items():
                    st.markdown(f"**{group_name}**")
                    cols = st.columns(len(group_vars))
                    for k, var in enumerate(group_vars):
                        with cols[k]:
                            inputs[var] = st.number_input(
                                var,
                                min_value=int(df[var].min()),
                                max_value=int(df[var].max()),
                                value=int(df[var].median()),
                                step=1,
                                help=f"Thang Likert {int(df[var].min())}–{int(df[var].max())}",
                            )
                submitted = st.form_submit_button("🔮 Dự báo", type="primary",
                                                  use_container_width=True)

            if submitted:
                X_new = pd.DataFrame([[inputs[v] for v in FEATURES]], columns=FEATURES)
                pred = int(model.predict(X_new)[0])
                proba = float(model.predict_proba(X_new)[0, 1])

                r1, r2 = st.columns(2)
                r1.metric("Kết quả dự báo (PD)", pred)
                r2.metric("Xác suất rủi ro (PD = 1)", f"{proba:.1%}")
                if pred == 1:
                    st.error("⚠️ Khách hàng được dự báo **CÓ RỦI RO** vỡ nợ.")
                else:
                    st.success("✅ Khách hàng được dự báo **KHÔNG CÓ RỦI RO** vỡ nợ.")

        # ---------- CHẾ ĐỘ 2: DỰ BÁO HÀNG LOẠT TỪ FILE ----------
        else:
            st.caption(
                "Tải lên tệp CSV có ĐÚNG 24 cột biến đầu vào: "
                + ", ".join(FEATURES)
            )
            batch_file = st.file_uploader(
                "Tải tệp dữ liệu cần dự báo (.csv)",
                type=["csv"],
                key="batch_uploader",
                help="Tệp phải chứa đủ 24 cột biến đầu vào giống cấu trúc X_test.",
            )
            if batch_file is not None:
                try:
                    df_new = pd.read_csv(io.BytesIO(batch_file.getvalue()),
                                         encoding="utf-8-sig")
                except Exception as e:
                    st.error(f"❌ Không đọc được tệp: {e}")
                    st.stop()

                if df_new.empty:
                    st.error("❌ Tệp dữ liệu rỗng.")
                else:
                    missing = validate_columns(df_new, FEATURES)
                    if missing:
                        st.error(
                            "❌ Tệp thiếu các cột bắt buộc: "
                            + ", ".join(f"`{c}`" for c in missing)
                        )
                    else:
                        extra = [c for c in df_new.columns if c not in FEATURES]
                        if extra:
                            st.warning(
                                "ℹ️ Các cột thừa sẽ được bỏ qua khi dự báo: "
                                + ", ".join(f"`{c}`" for c in extra)
                            )
                        try:
                            X_batch = df_new[FEATURES]
                            preds = model.predict(X_batch)
                            probas = model.predict_proba(X_batch)[:, 1]

                            out = df_new.copy()
                            out["PD_dự_báo"] = preds
                            out["Xác_suất_rủi_ro"] = probas

                            n_risk = int((preds == 1).sum())
                            b1, b2, b3 = st.columns(3)
                            b1.metric("Số quan sát", f"{len(out):,}")
                            b2.metric("Dự báo có rủi ro", f"{n_risk:,}")
                            b3.metric("Tỷ lệ rủi ro", f"{n_risk / len(out):.1%}")

                            with st.container(height=350):
                                st.dataframe(out, use_container_width=True)

                            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "⬇️ Tải kết quả dự báo (CSV)",
                                data=csv_bytes,
                                file_name="ket_qua_du_bao_5c.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"❌ Lỗi khi dự báo hàng loạt: {e}")
