import io

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

matplotlib.use("Agg")

from inference import analyze_image, load_class_mapping, load_prices, load_trained_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AISnack – Price Detector",
    page_icon="🍟",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .total-box {
        background: linear-gradient(135deg, #f58231, #e6194b);
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        color: white;
    }
    .total-label { font-size: 14px; opacity: 0.85; margin-bottom: 4px; }
    .total-value { font-size: 42px; font-weight: 800; letter-spacing: -1px; }
    .snack-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9a6324",
]


# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model = load_trained_model("models/snack_classifier.keras")
    class_mapping = load_class_mapping("config/classes.json")
    prices = load_prices("config/prices.json")
    return model, class_mapping, prices


# ── Annotation helper ─────────────────────────────────────────────────────────
def annotate_image(img_bgr: np.ndarray, result: dict) -> io.BytesIO:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_px, w_px = img_rgb.shape[:2]
    figw = 12
    figh = figw * h_px / w_px

    fig, ax = plt.subplots(1, figsize=(figw, figh))
    ax.imshow(img_rgb)
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    label_color: dict[str, str] = {}
    color_idx = 0

    for det in result["detections"]:
        label = det["label"]
        conf = det["confidence"]
        b = det["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]

        if label not in label_color:
            label_color[label] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
        color = label_color[label]

        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=max(1, w_px // 800),
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x + 4, y + 4,
            f"{label.replace('_', ' ')}  {conf:.0%}",
            color="white",
            fontsize=max(7, w_px // 300),
            fontweight="bold",
            va="top",
            bbox=dict(facecolor=color, alpha=0.82, pad=2, linewidth=0),
        )

    legend_handles = [
        mpatches.Patch(color=c, label=lbl.replace("_", " ").title())
        for lbl, c in label_color.items()
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=9,
            framealpha=0.88,
            title="Detected snacks",
        )

    n = len(result["detections"])
    total = result["total_price_rm"]
    ax.set_title(
        f"{n} detection{'s' if n != 1 else ''}  ·  Total: RM {total:.2f}",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=10,
    )
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍟 AISnack — Malaysian Snack Price Detector")
st.caption(
    "Upload a photo of Malaysian snacks and the app will identify each one "
    "and calculate the total price automatically."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection settings")
    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Minimum softmax score for a detection to be accepted.",
    )
    margin_thresh = st.slider(
        "Margin threshold",
        min_value=0.05, max_value=0.60, value=0.20, step=0.01,
        help="Minimum gap between the top-1 and top-2 class probabilities. "
             "A low margin means the model is uncertain.",
    )
    st.divider()
    st.markdown(
        "**Model**  \nMobileNetV2 + custom colour-based region proposals  \n"
        "**Classes**  \n22 Malaysian snack varieties  \n"
        "**Currency**  \nMalaysian Ringgit (RM)"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is None:
    st.info("👆 Upload a photo of snacks to get started.")
    st.stop()

model, class_mapping, prices = load_resources()

file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not decode the image. Please try a different file.")
    st.stop()

with st.spinner("Detecting snacks…"):
    result = analyze_image(
        img_bgr=img_bgr,
        model=model,
        class_mapping=class_mapping,
        prices=prices,
        confidence_threshold=conf_thresh,
        margin_threshold=margin_thresh,
    )

col_img, col_price = st.columns([3, 2], gap="large")

with col_img:
    st.subheader("📸 Detection result")
    buf = annotate_image(img_bgr, result)
    st.image(buf, use_container_width=True)

with col_price:
    st.subheader("🧾 Price breakdown")
    detections = result["detections"]

    if not detections:
        st.warning(
            "No snacks detected. Try lowering the **confidence threshold** in the sidebar."
        )
    else:
        breakdown = result["breakdown"]
        for label, info in breakdown.items():
            name = label.replace("_", " ").title()
            qty = info["count"]
            subtotal = info["subtotal_rm"]
            unit = info["unit_price_rm"]
            st.markdown(
                f"""
                <div class="snack-row">
                    <span>🍫 <b>{name}</b> × {qty}
                        <span style="color:#888; font-size:13px"> @ RM {unit:.2f}</span>
                    </span>
                    <span><b>RM {subtotal:.2f}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        total = result["total_price_rm"]
        st.markdown(
            f"""
            <div class="total-box">
                <div class="total-label">TOTAL PRICE</div>
                <div class="total-value">RM {total:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        with st.expander("Raw detection data"):
            for det in detections:
                st.json(
                    {
                        "label": det["label"],
                        "confidence": round(det["confidence"], 4),
                        "bbox": det["bbox"],
                    }
                )
