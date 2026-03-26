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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AISnack",
    page_icon="assets/icon.png" if (ROOT / "assets/icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    /* Page background */
    .stApp {
        background: linear-gradient(145deg, #EEF2FF 0%, #F8FAFC 55%, #EDF7FF 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid rgba(0, 0, 0, 0.07);
        box-shadow: 2px 0 24px rgba(0, 0, 0, 0.04);
    }

    /* App header banner */
    .app-header {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 122, 255, 0.25);
    }
    .app-header h1 {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -1px;
        margin: 0 0 4px 0 !important;
        padding: 0 !important;
    }
    .app-header p {
        color: rgba(255, 255, 255, 0.72);
        font-size: 15px;
        margin: 0;
    }

    /* Generic white card */
    .card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.04);
    }

    /* Image card — extra breathing room */
    .image-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 14px;
        margin-bottom: 22px;
        box-shadow: 0 6px 40px rgba(0, 0, 0, 0.09);
        border: 1px solid rgba(0, 0, 0, 0.04);
    }

    /* Stats chips row */
    .stats-bar {
        display: flex;
        gap: 10px;
        margin-bottom: 16px;
        flex-wrap: wrap;
        align-items: center;
    }
    .stat-chip {
        background: #FFFFFF;
        border-radius: 50px;
        padding: 8px 18px;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(0, 0, 0, 0.05);
        font-size: 14px;
        font-weight: 500;
        color: #1C1C1E;
    }
    .stat-chip-accent {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: #FFFFFF !important;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.30);
    }

    /* Section label */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #8E8E93;
        margin-bottom: 10px;
    }

    /* Price breakdown grid */
    .breakdown-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(165px, 1fr));
        gap: 12px;
        margin-bottom: 4px;
    }
    .breakdown-item {
        background: #F4F6FF;
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid rgba(0, 0, 0, 0.04);
    }
    .breakdown-item-name {
        font-size: 13px;
        font-weight: 700;
        color: #1C1C1E;
        margin-bottom: 3px;
    }
    .breakdown-item-meta {
        font-size: 12px;
        color: #8E8E93;
        margin-bottom: 10px;
    }
    .breakdown-item-price {
        font-size: 20px;
        font-weight: 800;
        color: #007AFF;
        letter-spacing: -0.4px;
    }

    /* Total card */
    .total-card {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 18px;
        padding: 22px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 14px;
        box-shadow: 0 8px 32px rgba(0, 122, 255, 0.28);
    }
    .total-card-label {
        font-size: 13px;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.75);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .total-card-value {
        font-size: 36px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -1.2px;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 8px;
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.06);
    }

    /* Sidebar info tiles */
    .info-tile {
        background: #F4F6FF;
        border-radius: 12px;
        padding: 11px 14px;
        margin-bottom: 8px;
    }
    .info-tile-label {
        font-size: 10px;
        color: #8E8E93;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .info-tile-value {
        font-size: 14px;
        color: #1C1C1E;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }

    hr {
        border: none;
        border-top: 1px solid #F0F0F0;
        margin: 16px 0;
    }
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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_px, w_px = img_rgb.shape[:2]

    # Wide figure so the image fills the full-width card clearly
    fig_w = 18
    fig_h = max(fig_w * h_px / w_px, 10)

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

        lw = max(2, w_px // 400)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        ))

        fs = max(9, w_px // 200)
        ax.text(
            x + lw + 2, y + lw + 2,
            f"{label.replace('_', ' ')}  {conf:.0%}",
            color="#FFFFFF",
            fontsize=fs,
            fontweight="semibold",
            va="top",
            bbox=dict(facecolor=color, alpha=0.92, pad=4, linewidth=0,
                      boxstyle="round,pad=0.35"),
        )

    if label_color:
        legend = [
            mpatches.Patch(color=c, label=lbl.replace("_", " ").title())
            for lbl, c in label_color.items()
        ]
        ax.legend(
            handles=legend,
            loc="upper right",
            fontsize=max(9, w_px // 250),
            framealpha=0.95,
            edgecolor="#E5E5EA",
            facecolor="#FFFFFF",
        )

    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>AISnack</h1>
        <p>Malaysian snack detection &amp; price estimation powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("<hr>", unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Minimum probability for a detection to be accepted.",
    )
    margin_thresh = st.slider(
        "Margin threshold",
        min_value=0.05, max_value=0.60, value=0.20, step=0.01,
        help="Minimum gap between the top two class probabilities. "
             "Low margin indicates an uncertain prediction.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Model info**")
    st.markdown(
        """
        <div class="info-tile">
            <div class="info-tile-label">Model</div>
            <div class="info-tile-value">MobileNetV2</div>
        </div>
        <div class="info-tile">
            <div class="info-tile-label">Proposals</div>
            <div class="info-tile-value">Colour-based (HSV)</div>
        </div>
        <div class="info-tile">
            <div class="info-tile-label">Classes</div>
            <div class="info-tile-value">22 varieties</div>
        </div>
        <div class="info-tile">
            <div class="info-tile-label">Currency</div>
            <div class="info-tile-value">Malaysian Ringgit (RM)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a snack image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:60px 24px;
             border: 2px dashed #D1D5E8; background:rgba(255,255,255,0.6);
             box-shadow:none;">
            <div style="font-size:52px; margin-bottom:16px;">📸</div>
            <div style="font-size:19px; font-weight:700; color:#1C1C1E; margin-bottom:8px;">
                Drop a snack photo here
            </div>
            <div style="font-size:14px; color:#8E8E93;">
                Supports JPG, PNG, and WebP · Use the uploader above
            </div>
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
total      = result["total_price_rm"]

# ── Stats bar ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="stats-bar">
        <div class="stat-chip">
            🍿&nbsp; <strong>{n_items}</strong>&nbsp;item{'s' if n_items != 1 else ''} detected
        </div>
        <div class="stat-chip stat-chip-accent">
            💰&nbsp; RM {total:.2f} total
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Detection image — full width ──────────────────────────────────────────────
st.markdown(
    '<div class="section-label">Detection result</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="image-card">', unsafe_allow_html=True)
buf = annotate_image(img_bgr, result)
st.image(buf, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Price breakdown — below the image ─────────────────────────────────────────
st.markdown(
    '<div class="section-label">Price breakdown</div>',
    unsafe_allow_html=True,
)

if not detections:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:48px 24px;">
            <div style="font-size:40px; margin-bottom:14px;">🔍</div>
            <div style="font-size:16px; font-weight:700; color:#1C1C1E; margin-bottom:6px;">
                No snacks detected
            </div>
            <div style="font-size:13px; color:#8E8E93;">
                Try lowering the Confidence slider in the sidebar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    items_html = ""
    for label, info in result["breakdown"].items():
        name     = label.replace("_", " ").title()
        qty      = info["count"]
        unit     = info["unit_price_rm"]
        subtotal = info["subtotal_rm"]
        items_html += f"""
        <div class="breakdown-item">
            <div class="breakdown-item-name">{name}</div>
            <div class="breakdown-item-meta">RM {unit:.2f} &times; {qty}</div>
            <div class="breakdown-item-price">RM {subtotal:.2f}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="card" style="padding:20px;">
            <div class="breakdown-grid">
                {items_html}
            </div>
        </div>
        <div class="total-card">
            <div class="total-card-label">Total</div>
            <div class="total-card-value">RM {total:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Raw detection data"):
        for det in detections:
            st.json({
                "label":      det["label"],
                "confidence": round(det["confidence"], 4),
                "bbox":       det["bbox"],
            })
