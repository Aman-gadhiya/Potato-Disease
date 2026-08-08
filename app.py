"""
=================================================================================
 POTATO LEAF DISEASE CLASSIFIER — STREAMLIT WEB APPLICATION
=================================================================================
 Project     : Potato Disease Classification (EfficientNetB0, Transfer Learning
               + Fine-Tuning)
 Purpose     : Let a user upload / capture a photo of a potato leaf and get an
               instant prediction of whether the leaf is healthy, or affected
               by Early Blight or Late Blight, along with the model's
               confidence and supporting evaluation evidence.
 Author      : Generated with Claude (Anthropic) from the user's trained model
               artifacts (potato_efficientnetb0_final.keras).
 How to run  : 1. pip install -r requirements.txt
               2. streamlit run app.py
=================================================================================
NOTE ON CODE STYLE:
This file is commented heavily — almost every line or small block explains
*why* it exists, not just *what* it does — so a beginner can follow along and
learn how a production-style ML inference app is put together.
=================================================================================
"""

# ---------------------------------------------------------------------------
# SECTION 0 — IMPORTS
# ---------------------------------------------------------------------------
# We keep imports grouped: standard library -> third-party -> local, which is
# the conventional (PEP 8-ish) ordering used in real-world Python projects.

import json                     # To read the small metadata files (class names, config) shipped with the model
import time                     # To measure how long inference takes, for the "prediction latency" readout
from pathlib import Path        # Object-oriented, OS-independent file path handling (safer than raw strings)

import numpy as np              # Numerical arrays — used to turn an image into the tensor the model expects
import pandas as pd             # DataFrames — used to display the evaluation tables (confusion matrix, metrics)
import streamlit as st          # The web app framework itself: turns this script into an interactive website
from PIL import Image           # Python Imaging Library — used to open, resize and validate uploaded images

# TensorFlow/Keras is imported lazily-ish (still at top level, but wrapped)
# because it is the heaviest import in the app and we want a clean error
# message if it is missing, instead of a confusing traceback.
try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - defensive import guard
    # If TensorFlow isn't installed, stop the app immediately with a clear,
    # actionable message rather than letting Streamlit crash with a raw
    # ModuleNotFoundError traceback that confuses non-technical users.
    st.error(
        "TensorFlow is not installed in this environment. "
        "Run `pip install -r requirements.txt` and restart the app."
    )
    st.stop()


# ---------------------------------------------------------------------------
# SECTION 1 — PAGE CONFIGURATION
# ---------------------------------------------------------------------------
# st.set_page_config() MUST be the very first Streamlit command executed in
# the script (before any st.write / st.title / etc.), otherwise Streamlit
# raises a runtime error. It controls the browser tab title, favicon, layout
# width, and the default state of the left sidebar.
st.set_page_config(
    page_title="Potato Disease Classifier",   # Text shown on the browser tab
    page_icon="🥔",                            # Emoji favicon (no external asset needed)
    layout="wide",                             # Use the full browser width instead of a narrow centered column
    initial_sidebar_state="expanded",          # Sidebar (navigation/info panel) starts open
)


# ---------------------------------------------------------------------------
# SECTION 2 — CONSTANTS & PATHS
# ---------------------------------------------------------------------------
# Centralising "magic values" like folder paths and thresholds at the top of
# the file means there is exactly ONE place to edit them later — a core
# software-engineering best practice that avoids bugs from inconsistent
# copies of the same value scattered through the code.

# Base directory that holds the model + its metadata files. Path(__file__)
# resolves to THIS script's location, so the app works no matter which
# directory the user launches `streamlit run` from.
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "model_assets"

MODEL_PATH = ASSETS_DIR / "potato_efficientnetb0_final.keras"          # The trained Keras model file
CLASS_MAP_PATH = ASSETS_DIR / "class_mapping.json"                     # index <-> class name lookup
PREPROCESS_CFG_PATH = ASSETS_DIR / "preprocessing_config.json"         # Expected input image shape
MODEL_METADATA_PATH = ASSETS_DIR / "model_metadata.json"               # High-level facts about the model
CLASS_METRICS_PATH = ASSETS_DIR / "final_test_class_metrics.csv"       # Per-class precision/recall/F1 on test set
CONFUSION_MATRIX_PATH = ASSETS_DIR / "final_confusion_matrix.csv"      # Confusion matrix on test set
TEST_SUMMARY_PATH = ASSETS_DIR / "final_test_evaluation_summary.csv"   # Overall test accuracy/loss/F1 summary

# Human-friendly display names + short care advice for each raw class label.
# Keeping this as a dictionary means the UI text is decoupled from the raw
# folder-name-style labels the model was trained on (e.g. "Potato___Early_blight").
DISPLAY_INFO = {
    "Potato___Early_blight": {
        "label": "Early Blight",
        "emoji": "🟤",
        "severity": "warning",
        "description": (
            "Early Blight is a fungal disease (Alternaria solani) that causes "
            "dark, concentric 'target-ring' spots on older leaves first."
        ),
        "advice": (
            "Remove and destroy infected leaves, avoid overhead watering, "
            "rotate crops yearly, and consider a labeled fungicide "
            "(e.g. chlorothalonil or copper-based) if the outbreak spreads. "
            "Consult a local agronomist for exact dosing."
        ),
    },
    "Potato___Late_blight": {
        "label": "Late Blight",
        "emoji": "🔴",
        "severity": "error",
        "description": (
            "Late Blight is caused by the oomycete Phytophthora infestans — "
            "the pathogen behind the historic Irish Potato Famine. It spreads "
            "fast in cool, wet weather and can destroy a field within days."
        ),
        "advice": (
            "Act quickly: remove infected plants, improve field drainage and "
            "airflow, and apply a recommended fungicide immediately. This "
            "disease is highly contagious to neighboring plants."
        ),
    },
    "Potato___healthy": {
        "label": "Healthy",
        "emoji": "🟢",
        "severity": "success",
        "description": "No visible signs of Early Blight or Late Blight were detected on this leaf.",
        "advice": (
            "Keep monitoring regularly, maintain good field hygiene, and "
            "continue balanced watering and fertilization to keep the crop healthy."
        ),
    },
}


# ---------------------------------------------------------------------------
# SECTION 3 — CACHED LOADERS
# ---------------------------------------------------------------------------
# @st.cache_resource tells Streamlit: "run this function once, keep the
# result (the loaded model) in memory, and reuse it on every future rerun /
# every user session — do NOT reload the 32 MB model file from disk on every
# button click." This is the single most important performance optimisation
# in a Streamlit ML app.
@st.cache_resource(show_spinner="Loading trained EfficientNetB0 model...")
def load_model(model_path: Path) -> tf.keras.Model:
    """Load the trained Keras model from disk exactly once and cache it."""
    # compile=False skips re-attaching the optimizer/loss (we only need the
    # model for inference/prediction, not further training), which loads faster.
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


# @st.cache_data is the sibling of @st.cache_resource, used for cacheable
# *data* (plain Python objects / DataFrames) rather than unpicklable objects
# like a TensorFlow model. It also avoids re-reading small JSON files
# repeatedly on every rerun.
@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    """Read and parse a small JSON metadata file."""
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV evaluation-artifact file into a pandas DataFrame."""
    # index_col=0 because these particular CSVs (e.g. the confusion matrix)
    # were exported with the row labels as the first column.
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# SECTION 4 — IMAGE PREPROCESSING & INFERENCE HELPERS
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """
    Convert a PIL image into the exact tensor shape/format the model expects.

    The training notebook builds `keras.applications.EfficientNetB0(...)`
    directly (via `image_dataset_from_directory`) WITHOUT a manual
    `/ 255.0` rescale step. Keras' built-in EfficientNet architecture embeds
    its own internal normalization layer, so the model expects RAW pixel
    values in the 0-255 range, not values pre-scaled to 0-1. Rescaling here
    a second time would silently corrupt every prediction, so we deliberately
    do NOT divide by 255.
    """
    # Force 3-channel RGB — this gracefully handles grayscale photos or PNGs
    # that include a 4th alpha (transparency) channel, both of which would
    # otherwise crash the model (which expects exactly 3 channels).
    rgb_image = pil_image.convert("RGB")

    # Resize to the model's expected input resolution (224 x 224 for
    # EfficientNetB0). Image.Resampling.BILINEAR gives a good quality/speed
    # trade-off for downscaling photos.
    resized_image = rgb_image.resize(target_size, Image.Resampling.BILINEAR)

    # Convert the PIL image to a NumPy array of dtype float32, matching the
    # "dtype": "float32" declared in preprocessing_config.json.
    image_array = np.asarray(resized_image, dtype=np.float32)

    # The model expects a *batch* of images, i.e. shape (batch, H, W, C).
    # A single image is (H, W, C), so we add a batch dimension of size 1
    # using np.expand_dims — this is one of the most common shape bugs
    # beginners hit when serving a single image to a Keras model.
    batched_array = np.expand_dims(image_array, axis=0)

    return batched_array


def run_inference(model: tf.keras.Model, image_batch: np.ndarray) -> np.ndarray:
    """Run a forward pass and return the raw softmax probability vector."""
    # verbose=0 suppresses Keras' progress bar printout, which would
    # otherwise clutter the Streamlit server logs on every prediction.
    predictions = model.predict(image_batch, verbose=0)
    # predictions has shape (1, num_classes) because of the batch dimension
    # we added earlier; [0] extracts the single prediction row we care about.
    return predictions[0]


def format_class_label(raw_label: str) -> str:
    """Map a raw training label (e.g. 'Potato___Early_blight') to display text."""
    return DISPLAY_INFO.get(raw_label, {}).get("label", raw_label)


# ---------------------------------------------------------------------------
# SECTION 5 — SIDEBAR (NAVIGATION + PROJECT INFO)
# ---------------------------------------------------------------------------
# Anything called on `st.sidebar` renders in the collapsible left panel
# instead of the main page body — ideal for navigation and secondary info
# that shouldn't compete for attention with the primary prediction workflow.
with st.sidebar:
    st.title("🥔 Potato Disease AI")
    st.caption("EfficientNetB0 · Transfer Learning + Fine-Tuning")

    # st.radio acts as simple page navigation without needing Streamlit's
    # separate multi-page-app file structure — perfect for a small app like
    # this one that has just a couple of views.
    page = st.radio(
        "Navigate",
        options=["🔍 Diagnose a Leaf", "📊 Model Performance", "ℹ️ About This Project"],
        label_visibility="collapsed",
    )

    st.divider()

    # Load metadata once here so we can show a quick trust-building summary
    # in the sidebar regardless of which page the user is on.
    model_metadata = load_json(MODEL_METADATA_PATH)

    st.markdown("**Quick facts**")
    st.markdown(
        f"- Architecture: `{model_metadata['model_name']}`\n"
        f"- Classes: `{model_metadata['num_classes']}`\n"
        f"- Best validation accuracy: `{model_metadata['best_validation_accuracy']:.2%}`\n"
        f"- Input size: `{model_metadata['input_size'][0]}x{model_metadata['input_size'][1]}`"
    )

    st.divider()
    st.caption(
        "⚠️ This tool assists field screening. It is not a certified "
        "agronomic diagnosis — confirm serious outbreaks with an expert."
    )


# ---------------------------------------------------------------------------
# SECTION 6 — PAGE: DIAGNOSE A LEAF (the core, primary feature)
# ---------------------------------------------------------------------------
if page == "🔍 Diagnose a Leaf":

    st.title("🥔 Potato Leaf Disease Classifier")
    st.markdown(
        "Upload a clear photo of a **single potato leaf** and the model will "
        "predict whether it is **healthy**, or shows signs of **Early Blight** "
        "or **Late Blight**."
    )

    # Load the heavy model + light config once (cached — see Section 3).
    model = load_model(MODEL_PATH)
    class_mapping = load_json(CLASS_MAP_PATH)
    preprocess_cfg = load_json(PREPROCESS_CFG_PATH)
    target_size = (preprocess_cfg["image_width"], preprocess_cfg["image_height"])

    # index_to_class in the JSON has string keys ("0", "1", "2") because JSON
    # object keys are always strings — we rebuild it with int keys so we can
    # index it directly with the model's predicted integer class index.
    index_to_class = {int(k): v for k, v in class_mapping["index_to_class"].items()}

    # st.columns splits the page into side-by-side regions — here, a left
    # column for image input/preview and a right column for results, which
    # mirrors how most real diagnostic tools lay out "input | result".
    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:
        st.subheader("1. Provide a leaf image")

        # Two intuitive input methods: uploading a file, or (on devices with
        # a camera, e.g. a phone in the field) capturing directly.
        input_method = st.radio(
            "Choose input method",
            options=["Upload a photo", "Use camera"],
            horizontal=True,
        )

        uploaded_image = None
        if input_method == "Upload a photo":
            uploaded_file = st.file_uploader(
                "Upload a leaf image (JPG / PNG)",
                type=["jpg", "jpeg", "png"],
                help="For best results, use a well-lit, close-up photo of a single leaf.",
            )
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)
        else:
            camera_file = st.camera_input("Take a photo of the leaf")
            if camera_file is not None:
                uploaded_image = Image.open(camera_file)

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Preview", use_container_width=True)

    with result_col:
        st.subheader("2. Prediction")

        if uploaded_image is None:
            # st.info renders a calm blue informational banner — appropriate
            # here because "no image yet" is an expected, non-error state.
            st.info("Upload or capture a leaf photo to see the prediction here.")
        else:
            # A spinner gives visual feedback while the model runs, so the
            # app doesn't feel frozen during the (usually sub-second) inference.
            with st.spinner("Analyzing leaf image..."):
                start_time = time.time()                                    # Start latency timer
                image_batch = preprocess_image(uploaded_image, target_size) # Resize/format for the model
                probabilities = run_inference(model, image_batch)           # Forward pass -> softmax scores
                elapsed_ms = (time.time() - start_time) * 1000              # Convert seconds -> milliseconds

            # np.argmax finds the index of the highest-probability class —
            # i.e. the model's single best guess.
            predicted_index = int(np.argmax(probabilities))
            predicted_raw_label = index_to_class[predicted_index]
            predicted_confidence = float(probabilities[predicted_index])
            display_info = DISPLAY_INFO[predicted_raw_label]

            # Render the headline result using the severity-appropriate
            # Streamlit banner (success = green, warning = yellow, error = red)
            # so the visual weight of the message matches how serious the
            # finding is — a UX detail that matters a lot in agri-tech tools.
            headline = f"{display_info['emoji']} **{display_info['label']}** ({predicted_confidence:.1%} confidence)"
            if display_info["severity"] == "success":
                st.success(headline)
            elif display_info["severity"] == "warning":
                st.warning(headline)
            else:
                st.error(headline)

            # A low-confidence prediction is worth flagging explicitly, since
            # blindly trusting a 40%-confidence guess could mislead a farmer.
            if predicted_confidence < 0.60:
                st.caption(
                    "⚠️ Confidence is relatively low — consider retaking the "
                    "photo with better lighting/focus, or getting a second opinion."
                )

            st.markdown(f"**What this means:** {display_info['description']}")
            st.markdown(f"**Suggested action:** {display_info['advice']}")

            st.divider()

            # Full probability breakdown across all 3 classes, not just the
            # winner — this transparency helps users judge borderline cases
            # (e.g. 55% Early Blight vs 45% Late Blight is very different
            # from a confident 98% call).
            st.markdown("**Full probability breakdown**")
            prob_df = pd.DataFrame(
                {
                    "Class": [format_class_label(index_to_class[i]) for i in range(len(probabilities))],
                    "Probability": probabilities,
                }
            ).sort_values("Probability", ascending=False)

            # st.bar_chart wants the category as the index, so we set it
            # explicitly before plotting.
            st.bar_chart(prob_df.set_index("Class"), horizontal=True)

            # Small technical footer — useful for debugging/demoing without
            # cluttering the main result above.
            st.caption(f"Inference time: {elapsed_ms:.0f} ms · Model input size: {target_size[0]}x{target_size[1]}")


# ---------------------------------------------------------------------------
# SECTION 7 — PAGE: MODEL PERFORMANCE (transparency / trust-building)
# ---------------------------------------------------------------------------
elif page == "📊 Model Performance":

    st.title("📊 Model Performance & Evaluation")
    st.markdown(
        "These numbers come directly from the held-out **test set** evaluation "
        "performed after training — i.e. data the model never saw during training."
    )

    # Load the three evaluation CSVs exported by the training pipeline.
    class_metrics_df = load_csv(CLASS_METRICS_PATH)
    confusion_df = load_csv(CONFUSION_MATRIX_PATH)
    test_summary_df = load_csv(TEST_SUMMARY_PATH)

    # --- Headline metrics as KPI-style cards -------------------------------
    # st.metric renders a large bold number with a small caption — the
    # standard way to surface top-line KPIs in a Streamlit dashboard.
    accuracy = float(test_summary_df.loc["Test Accuracy", "Value"])
    macro_f1 = float(test_summary_df.loc["Macro F1-Score", "Value"])
    total_samples = int(float(test_summary_df.loc["Total Test Samples", "Value"]))
    incorrect = int(float(test_summary_df.loc["Incorrect Predictions", "Value"]))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Test Accuracy", f"{accuracy:.2%}")
    kpi2.metric("Macro F1-Score", f"{macro_f1:.3f}")
    kpi3.metric("Test Samples", f"{total_samples}")
    kpi4.metric("Misclassified", f"{incorrect}", delta=f"-{incorrect}", delta_color="inverse")

    st.divider()

    # --- Per-class metrics table + chart ------------------------------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Per-class metrics")
        # Rename the raw training labels to friendly display names for the table.
        pretty_metrics = class_metrics_df.copy()
        pretty_metrics.index = [format_class_label(label) for label in pretty_metrics.index]
        st.dataframe(
            pretty_metrics.style.format("{:.3f}", subset=["Precision", "Recall", "F1-Score"]),
            use_container_width=True,
        )
        st.caption(
            "Precision = of all photos predicted as this class, how many truly were. "
            "Recall = of all real photos of this class, how many the model caught."
        )

    with right:
        st.subheader("Precision / Recall / F1 by class")
        chart_df = class_metrics_df[["Precision", "Recall", "F1-Score"]].copy()
        chart_df.index = [format_class_label(label) for label in chart_df.index]
        st.bar_chart(chart_df)

    st.divider()

    # --- Confusion matrix ---------------------------------------------------
    st.subheader("Confusion matrix")
    st.markdown(
        "Rows = the leaf's **true** class. Columns = what the model **predicted**. "
        "Values on the diagonal are correct predictions; anything off the "
        "diagonal is a mistake."
    )
    pretty_confusion = confusion_df.copy()
    pretty_confusion.index = [format_class_label(label) for label in pretty_confusion.index]
    pretty_confusion.columns = [format_class_label(label) for label in pretty_confusion.columns]

    # A background-gradient heatmap makes the diagonal (correct predictions)
    # visually pop out without needing a separate plotting library.
    st.dataframe(
        pretty_confusion.style.background_gradient(cmap="Greens", axis=None),
        use_container_width=True,
    )

    st.divider()

    # --- Raw summary table for full transparency -----------------------------
    with st.expander("See full raw evaluation summary"):
        st.dataframe(test_summary_df, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 8 — PAGE: ABOUT THIS PROJECT
# ---------------------------------------------------------------------------
else:  # page == "ℹ️ About This Project"

    st.title("ℹ️ About This Project")

    st.markdown(
        """
This application is powered by a **Convolutional Neural Network** built on
**EfficientNetB0**, trained using a two-stage **transfer learning + fine-tuning**
strategy to classify potato leaf photos into three categories:

- 🟢 **Healthy**
- 🟤 **Early Blight** (*Alternaria solani*)
- 🔴 **Late Blight** (*Phytophthora infestans*)
        """
    )

    st.subheader("Training pipeline summary")
    st.markdown(
        """
1. **Stage 1 — Feature extraction:** The EfficientNetB0 backbone (pretrained
   on ImageNet) was frozen, and only a new classification head was trained
   on the potato leaf dataset.
2. **Stage 2 — Fine-tuning:** A portion of the backbone was unfrozen and
   trained at a lower learning rate to adapt the pretrained visual features
   more specifically to potato leaf textures and lesion patterns.
3. **Evaluation:** The final model was benchmarked on a held-out test set
   that was never used for training or validation, to give an honest
   estimate of real-world performance (see the **Model Performance** page).
        """
    )

    model_metadata = load_json(MODEL_METADATA_PATH)
    with st.expander("Raw model metadata (model_metadata.json)"):
        st.json(model_metadata)

    st.subheader("Responsible use")
    st.warning(
        "This model was evaluated on a specific dataset and may perform "
        "differently on leaves photographed in different lighting, "
        "backgrounds, camera types, or growing regions than what it was "
        "trained on. Always validate uncertain or high-stakes cases with an "
        "agronomist before making treatment decisions."
    )

    st.subheader("Tech stack")
    st.markdown(
        """
- **Model:** TensorFlow / Keras — EfficientNetB0
- **Web app:** Streamlit
- **Image handling:** Pillow (PIL)
- **Data tables:** pandas
        """
    )
