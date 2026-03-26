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

# ── iOS-inspired design system ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue",
                     Arial, sans-serif;
    }

    /* Page background */
    .stApp {
        background-color: #F2F2F7;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5EA;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8E8E93;
        font-size: 13px;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* Section label */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8E8E93;
        margin-bottom: 12px;
    }

    /* Price row */
    .price-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 13px 0;
        border-bottom: 1px solid #F2F2F7;
    }
    .price-row:last-child {
        border-bottom: none;
    }
    .price-row-name {
        font-size: 15px;
        font-weight: 500;
        color: #1C1C1E;
    }
    .price-row-meta {
        font-size: 13px;
        color: #8E8E93;
        margin-top: 2px;
    }
    .price-row-value {
        font-size: 15px;
        font-weight: 600;
        color: #1C1C1E;
        white-space: nowrap;
    }

    /* Total card */
    .total-card {
        background: #007AFF;
        border-radius: 16px;
        padding: 22px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
    }
    .total-card-label {
        font-size: 15px;
        font-weight: 600;
        color: rgba(255,255,255,0.80);
        letter-spacing: 0.01em;
    }
    .total-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 8px;
    }

    /* Streamlit headings */
    h1 {
        font-weight: 700;
        font-size: 28px !important;
        color: #1C1C1E !important;
        letter-spacing: -0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 17px !important;
        color: #1C1C1E !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E5E5EA;
        margin: 20px 0;
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
@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    model       = load_trained_model(str(ROOT / "models" / "snack_classifier.h5"))
    class_map   = load_class_mapping(str(ROOT / "config" / "classes.json"))
    prices      = load_prices(str(ROOT / "config" / "prices.json"))
    return model, class_map, prices


# ── Annotation ────────────────────────────────────────────────────────────────
def annotate_image(img_bgr: np.ndarray, result: dict) -> io.BytesIO:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_px, w_px = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 12 * h_px / w_px))
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

        lw = max(2, w_px // 600)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        ))

        fs = max(8, w_px // 260)
        ax.text(
            x + lw + 2, y + lw + 2,
            f"{label.replace('_', ' ')}  {conf:.0%}",
            color="#FFFFFF",
            fontsize=fs,
            fontweight="semibold",
            va="top",
            bbox=dict(facecolor=color, alpha=0.90, pad=3, linewidth=0,
                      boxstyle="round,pad=0.3"),
        )

    if label_color:
        legend = [
            mpatches.Patch(color=c, label=lbl.replace("_", " ").title())
            for lbl, c in label_color.items()
        ]
        ax.legend(
            handles=legend,
            loc="upper right",
            fontsize=max(8, w_px // 300),
            framealpha=0.92,
            edgecolor="#E5E5EA",
            facecolor="#FFFFFF",
        )

    n     = len(result["detections"])
    total = result["total_price_rm"]
    ax.set_title(
        f"{n} item{'s' if n != 1 else ''} detected  —  RM {total:.2f}",
        fontsize=max(10, w_px // 250),
        fontweight="semibold",
        color="#1C1C1E",
        pad=12,
    )

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Header ────────────────────────────────────────────────────────────────────
st.title("AISnack")
st.caption("Malaysian snack detection and price estimation.")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    conf_thresh = st.slider(
        "Confidence",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Minimum probability for a detection to be accepted.",
    )
    margin_thresh = st.slider(
        "Margin",
        min_value=0.05, max_value=0.60, value=0.20, step=0.01,
        help="Minimum gap between the top two class probabilities. "
             "Low margin indicates an uncertain prediction.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "**Model**  \nMobileNetV2\n\n"
        "**Proposals**  \nColour-based (HSV)\n\n"
        "**Classes**  \n22 varieties\n\n"
        "**Currency**  \nMalaysian Ringgit (RM)"
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 48px 24px;">
            <div style="font-size:17px; font-weight:600; color:#1C1C1E; margin-bottom:8px;">
                No image selected
            </div>
            <div style="font-size:15px; color:#8E8E93;">
                Upload a photo using the button above.
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

with st.spinner("Analysing image..."):
    result = analyze_image(
        img_bgr=img_bgr,
        model=model,
        class_mapping=class_mapping,
        prices=prices,
        confidence_threshold=conf_thresh,
        margin_threshold=margin_thresh,
    )

# ── Results ───────────────────────────────────────────────────────────────────
col_img, col_price = st.columns([3, 2], gap="large")

with col_img:
    st.markdown('<div class="section-label">Detection result</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:12px;">', unsafe_allow_html=True)
    buf = annotate_image(img_bgr, result)
    st.image(buf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_price:
    st.markdown('<div class="section-label">Price breakdown</div>',
                unsafe_allow_html=True)

    detections = result["detections"]

    if not detections:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding:40px 24px;">
                <div style="font-size:15px; font-weight:600; color:#1C1C1E; margin-bottom:6px;">
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
        rows_html = ""
        for label, info in result["breakdown"].items():
            name     = label.replace("_", " ").title()
            qty      = info["count"]
            unit     = info["unit_price_rm"]
            subtotal = info["subtotal_rm"]
            rows_html += f"""
            <div class="price-row">
                <div>
                    <div class="price-row-name">{name}</div>
                    <div class="price-row-meta">RM {unit:.2f} x {qty}</div>
                </div>
                <div class="price-row-value">RM {subtotal:.2f}</div>
            </div>
            """

        total = result["total_price_rm"]
        st.markdown(
            f"""
            <div class="card">
                {rows_html}
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
