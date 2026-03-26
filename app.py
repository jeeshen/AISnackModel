import io
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

matplotlib.use("Agg")

from inference import analyze_image, load_class_mapping, load_prices, load_trained_model

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="AISnack",
    page_icon="assets/icon.png" if (ROOT / "assets/icon.png").exists() else None,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    *, html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Display",
                     "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        box-sizing: border-box;
    }

    /* ── Page ── */
    .stApp { background-color: #F2F2F7; }

    /* ── Remove Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 40px 24px 64px 24px !important;
        max-width: 780px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5EA;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 28px 20px;
    }

    /* ── App header ── */
    .app-header {
        margin-bottom: 6px;
    }
    .app-title {
        font-size: 36px;
        font-weight: 800;
        color: #1C1C1E;
        letter-spacing: -1.2px;
        line-height: 1;
        margin: 0;
    }
    .app-dot {
        color: #007AFF;
    }
    .app-subtitle {
        font-size: 15px;
        color: #8E8E93;
        margin-top: 6px;
        font-weight: 400;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background-color: #E5E5EA;
        border: none;
        margin: 20px 0;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #8E8E93;
        margin-bottom: 10px;
        padding-left: 2px;
    }

    /* ── Cards ── */
    .ios-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .img-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 14px;
        margin-bottom: 16px;
        /* ensure image takes full height */
        min-height: 480px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    /* Make Streamlit image inside img-card fill it */
    .img-card img {
        border-radius: 10px;
        width: 100%;
        height: auto;
        display: block;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 8px 12px;
    }

    /* ── Empty state ── */
    .empty-state {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 64px 24px;
        text-align: center;
        margin-top: 14px;
    }
    .empty-icon {
        font-size: 52px;
        line-height: 1;
        margin-bottom: 18px;
    }
    .empty-title {
        font-size: 17px;
        font-weight: 600;
        color: #1C1C1E;
        margin-bottom: 6px;
    }
    .empty-sub {
        font-size: 15px;
        color: #8E8E93;
    }

    /* ── Price rows ── */
    .price-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid #F2F2F7;
    }
    .price-row:last-child { border-bottom: none; }
    .price-item-name {
        font-size: 15px;
        font-weight: 500;
        color: #1C1C1E;
    }
    .price-item-meta {
        font-size: 13px;
        color: #8E8E93;
        margin-top: 2px;
    }
    .price-item-value {
        font-size: 15px;
        font-weight: 600;
        color: #1C1C1E;
        white-space: nowrap;
    }

    /* ── Total bar ── */
    .total-bar {
        background: #007AFF;
        border-radius: 20px;
        padding: 22px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    .total-label {
        font-size: 15px;
        font-weight: 600;
        color: rgba(255,255,255,0.80);
        letter-spacing: 0.01em;
    }
    .total-value {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.8px;
    }

    /* ── Detection badge ── */
    .det-badge {
        display: inline-flex;
        align-items: center;
        background: #E8F3FF;
        color: #007AFF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 5px 12px;
        border-radius: 100px;
        margin-bottom: 10px;
    }

    /* ── No detections state ── */
    .no-det-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 44px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .no-det-icon { font-size: 40px; margin-bottom: 12px; }
    .no-det-title {
        font-size: 15px;
        font-weight: 600;
        color: #1C1C1E;
        margin-bottom: 4px;
    }
    .no-det-sub { font-size: 13px; color: #8E8E93; }

    /* ── Sidebar styles ── */
    .sb-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #8E8E93;
        margin: 0 0 10px 0;
    }
    .sb-info-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 10px 0;
        border-bottom: 1px solid #F2F2F7;
    }
    .sb-info-row:last-child { border-bottom: none; }
    .sb-info-key {
        font-size: 14px;
        color: #8E8E93;
    }
    .sb-info-val {
        font-size: 14px;
        font-weight: 500;
        color: #1C1C1E;
    }

    /* ── Streamlit component overrides ── */
    h1, h2, h3 { color: #1C1C1E !important; }
    .stSlider [data-testid="stTickBar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour palette for bounding boxes ────────────────────────────────────────
PALETTE = [
    "#007AFF", "#34C759", "#FF9500", "#FF3B30",
    "#AF52DE", "#5AC8FA", "#FF2D55", "#FFCC00",
    "#30B0C7", "#32ADE6", "#64D2FF", "#BF5AF2",
]


# ── Model (cached across reruns) ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model     = load_trained_model(str(ROOT / "models" / "snack_classifier.h5"))
    class_map = load_class_mapping(str(ROOT / "config" / "classes.json"))
    prices    = load_prices(str(ROOT / "config" / "prices.json"))
    return model, class_map, prices


# ── Annotation ────────────────────────────────────────────────────────────────
def annotate_image(img_bgr: np.ndarray, result: dict) -> io.BytesIO:
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_px, w_px = img_rgb.shape[:2]

    # Keep aspect ratio; render large so text is crisp
    fig_w = 14
    fig_h = fig_w * h_px / w_px
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.imshow(img_rgb)
    ax.axis("off")

    label_color: dict[str, str] = {}
    color_idx = 0

    for det in result["detections"]:
        label = det["label"]
        conf  = det["confidence"]
        b     = det["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]

        if label not in label_color:
            label_color[label] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
        color = label_color[label]

        lw = max(2, w_px // 500)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        ))

        fs = max(9, w_px // 240)
        ax.text(
            x + lw + 2, y + lw + 2,
            f"{label.replace('_', ' ')}  {conf:.0%}",
            color="#FFFFFF",
            fontsize=fs,
            fontweight="semibold",
            va="top",
            bbox=dict(facecolor=color, alpha=0.92, pad=3.5,
                      linewidth=0, boxstyle="round,pad=0.3"),
        )

    if label_color:
        legend = [
            mpatches.Patch(color=c, label=lbl.replace("_", " ").title())
            for lbl, c in label_color.items()
        ]
        ax.legend(
            handles=legend,
            loc="upper right",
            fontsize=max(9, w_px // 280),
            framealpha=0.95,
            edgecolor="#E5E5EA",
            facecolor="#FFFFFF",
        )

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Minimum probability for a detection to be accepted.",
    )
    margin_thresh = st.slider(
        "Margin threshold",
        min_value=0.05, max_value=0.60, value=0.20, step=0.01,
        help="Minimum gap between the top two class probabilities.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#F2F2F7; border-radius:14px; padding:4px 14px;">
            <div class="sb-info-row">
                <span class="sb-info-key">Model</span>
                <span class="sb-info-val">MobileNetV2</span>
            </div>
            <div class="sb-info-row">
                <span class="sb-info-key">Proposals</span>
                <span class="sb-info-val">HSV colour</span>
            </div>
            <div class="sb-info-row">
                <span class="sb-info-key">Classes</span>
                <span class="sb-info-val">22 varieties</span>
            </div>
            <div class="sb-info-row">
                <span class="sb-info-key">Currency</span>
                <span class="sb-info-val">RM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">AI<span class="app-dot">Snack</span></div>
        <div class="app-subtitle">Malaysian snack detection &amp; price estimation</div>
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# ── Empty state ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📷</div>
            <div class="empty-title">No image selected</div>
            <div class="empty-sub">Upload a photo of Malaysian snacks above<br>to detect items and estimate the total price.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
model, class_mapping, prices = load_resources()

file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode the image. Please try a different file.")
    st.stop()

with st.spinner("Analysing image…"):
    result = analyze_image(
        img_bgr=img_bgr,
        model=model,
        class_mapping=class_mapping,
        prices=prices,
        confidence_threshold=conf_thresh,
        margin_threshold=margin_thresh,
    )

detections = result["detections"]
n_items    = len(detections)

# ── Detection result (full-width, tall image) ─────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)

if n_items > 0:
    st.markdown(
        f'<div class="det-badge">&#10003;&nbsp; {n_items} item{"s" if n_items != 1 else ""} detected</div>',
        unsafe_allow_html=True,
    )

buf = annotate_image(img_bgr, result)

st.markdown('<div class="img-card">', unsafe_allow_html=True)
st.image(buf, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Price breakdown ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Price breakdown</div>', unsafe_allow_html=True)

if not detections:
    st.markdown(
        """
        <div class="no-det-card">
            <div class="no-det-icon">🔍</div>
            <div class="no-det-title">No snacks detected</div>
            <div class="no-det-sub">Try lowering the Confidence threshold in the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    rows_html = ""
    for label, info in result["breakdown"].items():
        name     = label.replace("_", " ").title()
        qty      = info["count"]
        unit     = info["unit_price_rm"]
        subtotal = info["subtotal_rm"]
        rows_html += f"""
        <div class="price-row">
            <div>
                <div class="price-item-name">{name}</div>
                <div class="price-item-meta">RM {unit:.2f} × {qty}</div>
            </div>
            <div class="price-item-value">RM {subtotal:.2f}</div>
        </div>
        """

    total = result["total_price_rm"]
    st.markdown(
        f"""
        <div class="ios-card">
            {rows_html}
        </div>
        <div class="total-bar">
            <div class="total-label">Total</div>
            <div class="total-value">RM {total:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Raw detection data"):
        for det in detections:
            st.json({
                "label":      det["label"],
                "confidence": round(det["confidence"], 4),
                "bbox":       det["bbox"],
            })
