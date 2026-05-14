# ==========================================================
# 🎓 Personalized Learning Path Recommender (Error-Free, Clean UI)
# ==========================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# ---------- GEMINI CONFIG ----------
genai.configure(api_key="AIzaSyDNA0-0guf3d2Ol6prstPrD4WB8SXEihBQ")   

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("combined_dataset.csv")

df = load_data()

# ---------- TRAIN ML MODELS ----------
@st.cache_data
def train_models(df):
    FEATURE_COLS = [
        "attention_score",
        "avg_study_time_min",
        "quiz_score_pct",
        "prior_knowledge_level",
        "num_attempts",
        "engagement_events",
        "sleep_hours",
        "preferred_learning_style"
    ]
    TARGET_COL = "recommended_path"

    df_ml = df[FEATURE_COLS + [TARGET_COL]].dropna().reset_index(drop=True)

    # Encode categorical columns
    encoders = {}
    X = df_ml[FEATURE_COLS].copy()
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    target_enc = LabelEncoder()
    y = target_enc.fit_transform(df_ml[TARGET_COL].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    result = {
        "FEATURE_COLS": FEATURE_COLS,
        "encoders": encoders,
        "target_enc": target_enc,
        "dt": {
            "model": dt_model,
            "accuracy": accuracy_score(y_test, dt_pred),
            "report": classification_report(
                y_test, dt_pred, target_names=target_enc.classes_, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, dt_pred),
            "importances": getattr(dt_model, "feature_importances_", np.zeros(len(FEATURE_COLS))),
        },
        "rf": {
            "model": rf_model,
            "accuracy": accuracy_score(y_test, rf_pred),
            "report": classification_report(
                y_test, rf_pred, target_names=target_enc.classes_, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, rf_pred),
            "importances": getattr(rf_model, "feature_importances_", np.zeros(len(FEATURE_COLS))),
        },
    }
    return result

with st.spinner("Training models (Decision Tree + Random Forest)..."):
    ml_artifacts = train_models(df)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="🎓 Personalized Learning Path Recommender", layout="wide")
st.title("🎓 Personalized Learning Path Recommender Dashboard")
st.markdown("Empowering students through AI-driven personalized learning insights and recommendations.")

tab1, tab2 = st.tabs(["📊 Existing Student Insights", "🧠 Predict for New Student"])

# ==========================================================
# 📊 TAB 1 - EXISTING STUDENT INSIGHTS
# ==========================================================
with tab1:
    st.sidebar.header("📚 Select Student")
    student_index = st.sidebar.selectbox("Choose a student:", df.index)
    student_info = df.iloc[student_index]

    st.sidebar.write("### 🧾 Full Student Profile")
    for col in df.columns:
        st.sidebar.write(f"**{col.replace('_',' ').title()}:** {student_info.get(col, 'N/A')}")

    if st.button("✨ Generate Personalized Learning Path", key="tab1_btn"):
        try:
            prompt = f"""
            You are an AI tutor. Analyze this student's profile:
            {student_info.to_dict()}

            Suggest:
            - Learning priorities
            - Study plan (daily/weekly)
            - Improvement areas
            - Motivational suggestions
            """
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content(prompt)
            st.success("✅ Recommendation Generated Successfully!")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini Error: {e}")

    st.markdown("---")
    st.header("📈 Data Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Learning Style Distribution")
        st.plotly_chart(px.pie(df, names="preferred_learning_style", title="Learning Styles"),
                        use_container_width=True)
    with col2:
        st.subheader("Study Time by Knowledge Level")
        st.plotly_chart(px.bar(df, x="prior_knowledge_level", y="avg_study_time_min",
                               color="preferred_learning_style", text_auto=True),
                        use_container_width=True)

    st.subheader("Attention vs Quiz Score")
    st.plotly_chart(px.scatter(df, x="attention_score", y="quiz_score_pct",
                               color="preferred_learning_style", hover_data=["sleep_hours"]),
                    use_container_width=True)

    st.markdown("---")
    st.header("🤖 Model Summary")

    # 🔸 Side-by-side layout for Decision Tree and Random Forest
    dt = ml_artifacts["dt"]
    rf = ml_artifacts["rf"]

    col_dt, col_rf = st.columns(2)
    with col_dt:
        st.subheader("🌳 Decision Tree")
        st.metric("Accuracy", f"{dt['accuracy']:.3f}")
        st.text(dt["report"])
    with col_rf:
        st.subheader("🌲 Random Forest")
        st.metric("Accuracy", f"{rf['accuracy']:.3f}")
        st.text(rf["report"])

# ==========================================================
# 🧠 TAB 2 - NEW STUDENT PREDICTION (Fixed Encoding + Clean UI)
# ==========================================================
with tab2:
    st.header("🧩 Predict for a New Student")

    with st.form("predict_form"):
        attention_score = st.slider("Attention Score", 0, 100, 70)
        avg_study_time_min = st.number_input("Average Study Time (min)", 0, 600, 120)
        quiz_score_pct = st.slider("Quiz Score (%)", 0, 100, 75)
        prior_knowledge_level = st.selectbox("Prior Knowledge", ["Low", "Medium", "High"])
        num_attempts = st.number_input("Number of Attempts", 0, 20, 3)
        engagement_events = st.number_input("Engagement Events", 0, 200, 50)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        preferred_learning_style = st.selectbox("Preferred Learning Style",
                                                ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"])
        recommended_path = st.text_input("Current Recommended Path", "")
        suggested_topic_focus = st.text_input("Suggested Topic Focus", "")
        submit = st.form_submit_button("🔍 Predict & Generate Path")

    if submit:
        sample_df = pd.DataFrame([{
            "attention_score": attention_score,
            "avg_study_time_min": avg_study_time_min,
            "quiz_score_pct": quiz_score_pct,
            "prior_knowledge_level": prior_knowledge_level,
            "num_attempts": num_attempts,
            "engagement_events": engagement_events,
            "sleep_hours": sleep_hours,
            "preferred_learning_style": preferred_learning_style
        }])

        # --- FIXED ENCODING STEP ---
        encoders = ml_artifacts["encoders"]
        for col in sample_df.columns:
            if col in encoders:
                le = encoders[col]
                # Encode safely even if unseen category appears
                try:
                    sample_df[col] = le.transform(sample_df[col])
                except ValueError:
                    sample_df[col] = le.transform([le.classes_[0]])[0]

        # Ensure numeric dtype now (safe)
        X_input = sample_df[ml_artifacts["FEATURE_COLS"]].apply(pd.to_numeric, errors='coerce').fillna(0)

        dt_model = ml_artifacts["dt"]["model"]
        rf_model = ml_artifacts["rf"]["model"]
        target_enc = ml_artifacts["target_enc"]

        try:
            dt_pred = dt_model.predict(X_input)
            dt_label = target_enc.inverse_transform(dt_pred)[0]
        except Exception:
            dt_label = "N/A"

        try:
            rf_pred = rf_model.predict(X_input)
            rf_label = target_enc.inverse_transform(rf_pred)[0]
        except Exception:
            rf_label = "N/A"

        # ⚡ Use predictions backend only (no UI display)
        try:
            prompt = f"""
            Student Data:
            {sample_df.to_dict(orient='records')[0]}

            Decision Tree predicted: {dt_label}
            Random Forest predicted: {rf_label}

            Based on this, generate a personalized learning plan including:
            - Study routine
            - Priority topics
            - Best learning methods
            - Motivation tip
            """
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content(prompt)
            st.success("✅ Personalized Learning Path Generated!")
            st.write(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini Error: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("MINI PROJECT")
