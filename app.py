import streamlit as st
import pickle
import string
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Language Detector",
    page_icon="🌍",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["English", "Swahili", "Sheng", "Kikuyu"]

LANGUAGE_FLAGS = {
    "English":  "🇬🇧",
    "Swahili":  "🇰🇪",
    "Sheng":    "🗣️",
    "Kikuyu":   "🌿",
}

# ── Model loading (cached so it only runs once) ───────────────────────────────
@st.cache_resource
def load_model():
    """Load model and vectorizer from disk. Returns (model, vectorizer) or raises."""
    model_path      = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    missing = [p for p in [model_path, vectorizer_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s): {', '.join(missing)}. "
            "Make sure model.pkl and vectorizer.pkl are in the same directory as this script."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# ── Text preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase and strip punctuation — must match training preprocessing."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_language(text: str, model, vectorizer) -> dict:
    """
    Returns a dict with:
        label       – top predicted language (str)
        confidence  – probability of top label (float | None)
        proba_map   – {language: probability} for all classes (dict | None)
    """
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    label      = model.predict(vectorized)[0]

    proba_map = None
    confidence = None
    if hasattr(model, "predict_proba"):
        proba          = model.predict_proba(vectorized)[0]
        classes        = list(model.classes_)
        proba_map      = dict(zip(classes, proba))
        confidence     = proba_map.get(label, None)

    return {"label": label, "confidence": confidence, "proba_map": proba_map}


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🌍 Language Detection")
    st.caption("Detects: English · Swahili · Sheng · Kikuyu")
    st.divider()

    # Load model
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Input
    user_input = st.text_area(
        "Enter text to analyse",
        placeholder="Type or paste text here…",
        height=140,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        predict_btn = st.button("Detect", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # Prediction
    if predict_btn:
        text = user_input.strip()

        if not text:
            st.warning("⚠️ Please enter some text before clicking Detect.")
            return

        if len(text.split()) < 2:
            st.info("💡 For better accuracy, enter at least a couple of words.")

        try:
            result = predict_language(text, model, vectorizer)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        label      = result["label"]
        confidence = result["confidence"]
        proba_map  = result["proba_map"]
        flag       = LANGUAGE_FLAGS.get(label, "🌐")

        st.divider()

        # Main result card
        conf_str = f"  —  {confidence:.0%} confidence" if confidence is not None else ""
        st.success(f"{flag}  **{label}** detected{conf_str}")

        # Confidence bar chart (if model supports probabilities)
        if proba_map:
            st.markdown("**Confidence breakdown**")
            # Sort by probability descending
            sorted_proba = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)
            for lang, prob in sorted_proba:
                bar_label = f"{LANGUAGE_FLAGS.get(lang, '')} {lang}"
                st.progress(float(prob), text=f"{bar_label}  {prob:.1%}")

        # Word & character stats
        with st.expander("📊 Input statistics"):
            words = len(text.split())
            chars = len(text)
            unique = len(set(text.lower().split()))
            c1, c2, c3 = st.columns(3)
            c1.metric("Words",          words)
            c2.metric("Characters",     chars)
            c3.metric("Unique words",   unique)


if __name__ == "__main__":
    main()