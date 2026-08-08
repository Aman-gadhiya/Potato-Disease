# 🥔 Potato Leaf Disease Classifier — Streamlit App

A production-style web app that uses your trained **EfficientNetB0** model
(transfer learning + fine-tuning) to classify potato leaf photos as
**Healthy**, **Early Blight**, or **Late Blight**.

## Project structure

```
app/
├── app.py                        # The Streamlit application (heavily commented)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── model_assets/
    ├── potato_efficientnetb0_final.keras   # Trained model (from your production folder)
    ├── class_mapping.json                  # index <-> class name mapping
    ├── preprocessing_config.json           # Expected input image size/dtype
    ├── model_metadata.json                 # High-level model facts (shown in the "About" page)
    ├── final_test_class_metrics.csv        # Per-class precision/recall/F1 on the test set
    ├── final_confusion_matrix.csv          # Confusion matrix on the test set
    └── final_test_evaluation_summary.csv   # Overall accuracy/loss/F1 summary
```

## Setup

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## What the app does

1. **🔍 Diagnose a Leaf** — Upload or capture a photo of a potato leaf. The
   app resizes/formats it exactly the way the model was trained on, runs
   inference, and shows the predicted class, confidence score, a full
   probability breakdown across all 3 classes, and simple care guidance.
2. **📊 Model Performance** — Shows the model's real test-set results
   (accuracy, macro F1, per-class precision/recall/F1, and a confusion
   matrix), sourced directly from your training pipeline's exported CSVs.
3. **ℹ️ About This Project** — Explains the two-stage transfer learning +
   fine-tuning approach and includes responsible-use guidance.

## Important implementation detail: preprocessing

Your training notebook builds the model directly from
`keras.applications.EfficientNetB0(include_top=False, weights="imagenet")`
and feeds it images from `image_dataset_from_directory` **without** a manual
`/ 255.0` rescale step. Keras' EfficientNet family has rescaling/normalization
built into the model graph itself, so this app intentionally sends the model
raw `0–255` pixel values (converted to `float32`) — **not** pre-normalized
`0–1` values. If you ever retrain with a different preprocessing scheme,
update `preprocess_image()` in `app.py` to match.

## Deploying

This app has no external network or API dependencies, so it deploys as-is to
[Streamlit Community Cloud](https://streamlit.io/cloud), Hugging Face
Spaces, or any server that can run `streamlit run app.py`. Just make sure the
`model_assets/` folder is included in your deployment — the whole app is
~35 MB and needs no GPU (EfficientNetB0 inference is fast on CPU).
