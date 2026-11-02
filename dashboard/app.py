
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import select

from app.database.models import Experiment, EvaluationResult, get_session_factory


st.set_page_config(
    page_title="Persian AI Benchmark",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DARK PROFESSIONAL THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

    :root {
      --bg: #080d19;
      --panel: #111a2b;
      --panel-2: #172338;
      --border: rgba(148, 163, 184, .16);
      --text: #e7eefb;
      --muted: #91a2bd;
      --cyan: #35d6d0;
      --blue: #5798ff;
      --purple: #a78bfa;
      --green: #4ade80;
      --orange: #fbbf24;
      --pink: #f472b6;
    }

    html, body, [class*="css"],
    [data-testid="stAppViewContainer"] {
      font-family: 'Vazirmatn', sans-serif;
      direction: rtl;
      text-align: right;
    }

    [data-testid="stAppViewContainer"] {
      background: radial-gradient(
        ellipse at 12% 0%,
        #142744 0%,
        var(--bg) 43%
      );
      color: var(--text);
    }

    [data-testid="stHeader"] {
      background: rgba(8,13,25,.78);
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(
        180deg,
        #101a2b 0%,
        #0b1220 100%
      );
      border-left: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
      font-family: 'Vazirmatn', sans-serif;
    }

    [data-testid="stSidebarNav"] {
      direction: rtl;
    }

    .block-container {
      padding-top: 2rem;
      padding-bottom: 3rem;
      max-width: 1500px;
    }

    h1, h2, h3, p, label {
      color: var(--text);
    }

    h1 {
      font-weight: 800 !important;
      letter-spacing: -.5px;
    }

    h2, h3 {
      font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] p,
    .muted {
      color: var(--muted) !important;
    }

    .hero {
      padding: 1.45rem 1.6rem;
      margin-bottom: 1.4rem;
      border: 1px solid rgba(53,214,208,.22);
      border-radius: 20px;
      background: linear-gradient(
        115deg,
        rgba(25,55,83,.92),
        rgba(17,26,43,.94) 62%,
        rgba(50,39,83,.72)
      );
      box-shadow: 0 16px 45px rgba(0,0,0,.18);
    }

    .hero-title {
      font-size: 1.8rem;
      font-weight: 800;
      color: #f4f8ff;
      margin: 0 0 .35rem 0;
    }

    .hero-subtitle {
      color: #a9bad3;
      font-size: .94rem;
      margin: 0;
    }

    .eyebrow {
      color: var(--cyan);
      font-size: .76rem;
      font-weight: 700;
      letter-spacing: 1px;
      margin-bottom: .4rem;
    }

    .section-heading {
      color: #e8f0ff;
      font-size: 1.15rem;
      font-weight: 700;
      margin: 1.5rem 0 .9rem;
      padding-right: .7rem;
      border-right: 3px solid var(--cyan);
    }

    .kpi-card {
      min-height: 132px;
      padding: 1.05rem 1.15rem;
      background: linear-gradient(
        145deg,
        rgba(24,38,60,.98),
        rgba(15,24,40,.98)
      );
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgba(0,0,0,.16);
    }

    .kpi-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .kpi-label {
      color: #9fb0c9;
      font-size: .82rem;
      font-weight: 500;
    }

    .kpi-icon {
      width: 34px;
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 11px;
      font-size: 1rem;
      background: rgba(53,214,208,.12);
    }

    .kpi-value {
      color: #f5f8ff;
      font-size: 1.75rem;
      font-weight: 800;
      direction: ltr;
      text-align: right;
      margin-top: .45rem;
    }

    .kpi-foot {
      color: #8294b0;
      font-size: .72rem;
      margin-top: .1rem;
    }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stRadio"] > label {
      color: #c9d6e9;
      font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
      background: #172338;
      border-color: #2b3c56;
      color: #edf4ff;
      border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }

    div[data-testid="stExpander"] {
      background: rgba(20,32,51,.8);
      border: 1px solid var(--border);
      border-radius: 12px;
    }

    div[data-testid="stAlert"] {
      border-radius: 12px;
    }

    .stButton button {
      border-radius: 10px;
    }

    hr {
      border-color: var(--border);
    }

    footer {
      visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

Session = get_session_factory()

with Session() as session:
    experiments = session.scalars(
        select(Experiment).order_by(Experiment.id.desc())
    ).all()

    rows = [
        {
            "شناسه": e.id,
            "مدل": e.model,
            "ارائه‌دهنده": e.provider,
            "تعداد پرسش": e.total_questions,
            "ارتباط پاسخ": e.answer_relevancy,
            "وفاداری به زمینه": e.faithfulness,
            "نرخ توهم": e.hallucination_rate,
            "Recall@K": e.recall_at_k,
            "MRR": e.mrr,
            "تأخیر (ms)": e.average_latency_ms,
            "هزینه تخمینی ($)": e.estimated_cost_usd,
            "تاریخ": e.created_at,
        }
        for e in experiments
    ]

frame = pd.DataFrame(rows)

if frame.empty:
    total_questions = 0
    avg_quality = 0.0
    avg_latency = 0.0
    total_cost = 0.0
    avg_recall = 0.0
    avg_mrr = 0.0
else:
    total_questions = int(frame["تعداد پرسش"].sum())
    avg_quality = float(frame["ارتباط پاسخ"].mean())
    avg_latency = float(frame["تأخیر (ms)"].mean())
    total_cost = float(frame["هزینه تخمینی ($)"].sum())
    avg_recall = float(frame["Recall@K"].mean())
    avg_mrr = float(frame["MRR"].mean())


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI QUALITY & RETRIEVAL ANALYTICS</div>
      <div class="hero-title">
        🧪 پلتفرم ارزیابی هوش مصنوعی فارسی
      </div>
      <p class="hero-subtitle">
        مرکز پایش کیفیت پاسخ، بازیابی اطلاعات و مقایسه
        آزمایش‌های مدل‌های زبانی و سامانه‌های RAG
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">نمای کلی عملکرد</div>',
    unsafe_allow_html=True,
)

kpis = [
    ("کل آزمایش‌ها", f"{len(frame):,}", "آزمایش ثبت‌شده", "🧪"),
    ("پرسش‌های ارزیابی‌شده", f"{total_questions:,}", "مجموع پرسش‌ها", "📚"),
    ("میانگین ارتباط پاسخ", f"{avg_quality:.2f}", "Answer Relevancy", "✦"),
    ("میانگین تأخیر", f"{avg_latency:.2f} ms", "زمان گزارش‌شده", "⚡"),
    ("هزینه تخمینی", f"${total_cost:.4f}", "مجموع هزینه آزمایش‌ها", "＄"),
]

cols = st.columns(5)

for col, (label, value, foot, icon) in zip(cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-top">
                <span class="kpi-label">{label}</span>
                <span class="kpi-icon">{icon}</span>
              </div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-foot">{foot}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🧭 ناوبری")

page = st.sidebar.radio(
    "بخش موردنظر را انتخاب کنید",
    [
        "نمای کلی و مقایسه",
        "ارزیابی بازیابی",
        "تاریخچه آزمایش‌ها",
        "نتایج تفصیلی",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown("### وضعیت داده‌ها")
st.sidebar.markdown(f"**تعداد آزمایش‌ها:** `{len(frame)}`")
st.sidebar.markdown(f"**تعداد پرسش‌ها:** `{total_questions}`")
st.sidebar.caption(
    "داده‌ها مستقیماً از پایگاه داده محلی خوانده می‌شوند."
)


# --------------------------------------------------
# CHART CONFIGURATION
# --------------------------------------------------

PLOT_COLORS = [
    "#35d6d0",
    "#5798ff",
    "#a78bfa",
    "#4ade80",
    "#fbbf24",
    "#f472b6",
]

PLOT_TEMPLATE = "plotly_dark"


def style_figure(fig, height=390):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Vazirmatn, sans-serif",
            "color": "#cbd7e8",
            "size": 12,
        },
        margin={
            "l": 18,
            "r": 18,
            "t": 45,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 1,
            "xanchor": "right",
        },
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,.18)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.13)",
        zeroline=False,
    )

    return fig


# --------------------------------------------------
# DASHBOARD PAGES
# --------------------------------------------------

if frame.empty:

    st.info(
        "هنوز آزمایشی ثبت نشده است. ابتدا یک Benchmark اجرا کنید."
    )

else:

    # PAGE 1: OVERVIEW

    if page == "نمای کلی و مقایسه":

        st.markdown(
            '<div class="section-heading">مقایسه کیفیت و عملکرد مدل‌ها</div>',
            unsafe_allow_html=True,
        )

        metric = st.selectbox(
            "معیار موردنظر",
            [
                "ارتباط پاسخ",
                "وفاداری به زمینه",
                "نرخ توهم",
                "Recall@K",
                "MRR",
                "تأخیر (ms)",
                "هزینه تخمینی ($)",
            ],
        )

        fig = px.bar(
            frame,
            x="مدل",
            y=metric,
            color="ارائه‌دهنده",
            barmode="group",
            color_discrete_sequence=PLOT_COLORS,
            text_auto=".3f",
        )

        fig.update_traces(
            marker_line_width=0,
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(
            style_figure(fig),
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.markdown(
            '<div class="section-heading">خلاصه آزمایش‌ها</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )


    # PAGE 2: RETRIEVAL

    elif page == "ارزیابی بازیابی":

        st.markdown(
            '<div class="section-heading">کیفیت بازیابی اطلاعات</div>',
            unsafe_allow_html=True,
        )

        retrieval = frame.melt(
            id_vars=["مدل", "شناسه"],
            value_vars=["Recall@K", "MRR"],
            var_name="معیار",
            value_name="امتیاز",
        )

        fig = px.bar(
            retrieval,
            x="مدل",
            y="امتیاز",
            color="معیار",
            barmode="group",
            color_discrete_sequence=[
                "#35d6d0",
                "#a78bfa",
            ],
            text_auto=".3f",
        )

        fig.update_traces(
            marker_line_width=0,
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(
            style_figure(fig),
            use_container_width=True,
            config={"displaylogo": False},
        )

        a, b = st.columns(2)

        a.metric(
            "میانگین Recall@K",
            f"{avg_recall:.3f}",
        )

        b.metric(
            "میانگین MRR",
            f"{avg_mrr:.3f}",
        )


    # PAGE 3: EXPERIMENT HISTORY

    elif page == "تاریخچه آزمایش‌ها":

        st.markdown(
            '<div class="section-heading">تاریخچه آزمایش‌ها</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )

        if len(frame) >= 2:

            st.markdown(
                '<div class="section-heading">مقایسه دو آزمایش</div>',
                unsafe_allow_html=True,
            )

            ids = frame["شناسه"].tolist()

            left, right = st.columns(2)

            first = left.selectbox(
                "آزمایش اول",
                ids,
                key="history_a",
            )

            second = right.selectbox(
                "آزمایش دوم",
                ids,
                index=1,
                key="history_b",
            )

            st.dataframe(
                frame[
                    frame["شناسه"].isin([first, second])
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "برای مقایسه روند، حداقل دو آزمایش باید ثبت شده باشد."
            )


    # PAGE 4: DETAILED RESULTS

    else:

        st.markdown(
            '<div class="section-heading">نتایج تفصیلی پرسش‌ها</div>',
            unsafe_allow_html=True,
        )

        selected = st.selectbox(
            "شناسه آزمایش",
            frame["شناسه"].tolist(),
        )

        with Session() as session:

            results = session.scalars(
                select(EvaluationResult).where(
                    EvaluationResult.experiment_id == selected
                )
            ).all()

            details = [
                {
                    "پرسش": r.question,
                    "پاسخ مرجع": r.reference_answer,
                    "پاسخ مدل": r.generated_answer,
                    "زمینه بازیابی‌شده": r.retrieved_context,
                    "Relevancy": r.relevancy,
                    "Faithfulness": r.faithfulness,
                    "Recall@K": r.recall_at_k,
                    "MRR": r.mrr,
                }
                for r in results
            ]

        if not details:

            st.info(
                "برای این آزمایش نتیجه تفصیلی ثبت نشده است."
            )

        else:

            st.caption(
                f"{len(details)} نتیجه برای این آزمایش"
            )

            for item in details:

                with st.expander(item["پرسش"]):

                    for key, value in item.items():

                        if key != "پرسش":

                            st.markdown(f"**{key}**")

                            st.write(
                                value
                                if value not in (None, "")
                                else "—"
                            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Persian AI Benchmark · داشبورد ارزیابی محلی مدل‌ها و سامانه‌های RAG"
)
