"""
Veritas AI - Fake News Detection & Credibility Analysis Platform.
Interactive Streamlit Web Application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import FakeNewsModelManager
from src.heuristics import analyze_linguistic_features
from src.preprocessor import clean_text

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Veritas AI | Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Sleek card containers */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        backdrop-filter: blur(8px);
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        animation: pulseFake 2s infinite;
    }
    
    .verdict-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(21, 128, 61, 0.1) 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .tag-danger {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .tag-success {
        background-color: rgba(34, 197, 94, 0.25);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    .feature-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management & Caching
# ---------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "news_dataset.csv")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_articles.json")

@st.cache_resource
def get_model_manager(data_version: int = 1):
    if not os.path.exists(DATA_PATH):
        from generate_dataset import build_dataset
        build_dataset()
        
    df = pd.read_csv(DATA_PATH)
    manager = FakeNewsModelManager(model_type="Ensemble")
    metrics = manager.train_all(df)
    return manager, df

if "data_version" not in st.session_state:
    st.session_state.data_version = 1

model_manager, current_df = get_model_manager(st.session_state.data_version)

# Load sample articles
samples = []
if os.path.exists(SAMPLES_PATH):
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **Veritas AI**")
    st.caption("Machine Learning Fake News Detection & Credibility Analysis")
    
    st.markdown("---")
    st.markdown("#### **Active ML Engine**")
    selected_model = st.selectbox(
        "Classifier Algorithm",
        ["Ensemble (Recommended)", "PassiveAggressive", "LogisticRegression", "NaiveBayes", "LinearSVC"],
        index=0,
        help="Ensemble aggregates predictions across all models for maximum stability."
    )
    clean_model_name = "Ensemble" if "Ensemble" in selected_model else selected_model
    
    st.markdown("---")
    st.markdown("#### **Corpus Statistics**")
    real_count = int((current_df["label"] == "REAL").sum())
    fake_count = int((current_df["label"] == "FAKE").sum())
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Real Articles", real_count)
    with col_sb2:
        st.metric("Fake Articles", fake_count)
        
    st.metric("Total Vocabulary Size", f"{len(model_manager.vectorizer.get_feature_names_out()):,} n-grams")
    
    st.markdown("---")
    st.markdown("#### **Quick Guide**")
    st.caption("• **Passive-Aggressive**: Fast online margin-based classifier ideal for streaming text.")
    st.caption("• **Logistic Regression**: Calibrated probabilities with L2 regularization.")
    st.caption("• **Sensationalism Index**: Stylistic scoring based on caps, punctuation, and triggers.")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("🛡️ Fake News Detection & NLP Verification")
st.markdown(
    "Detect deceptive information, propaganda, clickbait headlines, and pseudo-scientific falsehoods "
    "using **TF-IDF N-Gram Vectorization**, **Calibrated Multi-Model Ensembles**, and **Stylistic Heuristics**."
)

tab_detect, tab_dataset, tab_performance, tab_upload = st.tabs([
    "🔍 Live News Detector",
    "📊 Dataset Explorer",
    "⚙️ Model Performance & Benchmarks",
    "📁 Upload Custom Dataset"
])

# =========================================================
# TAB 1: LIVE NEWS DETECTOR
# =========================================================
with tab_detect:
    st.markdown("### Test News Content")
    
    # Preset Article Loader
    sample_options = ["Custom Input (Type/Paste your own)"] + [f"{s['expected']}: {s['name']}" for s in samples]
    selected_sample = st.selectbox("⚡ Quick-Fill with Verified Test Case:", sample_options, index=0)
    
    preset_title = ""
    preset_text = ""
    if selected_sample != "Custom Input (Type/Paste your own)":
        idx = sample_options.index(selected_sample) - 1
        preset_title = samples[idx]["title"]
        preset_text = samples[idx]["text"]
        
    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        input_title = st.text_input("News Headline / Title:", value=preset_title, placeholder="e.g. Breaking: New study finds...")
    with col_in2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Analyze Credibility", type="primary", use_container_width=True)

    input_text = st.text_area(
        "Article Body / Excerpt (Optional but improves accuracy):",
        value=preset_text,
        height=150,
        placeholder="Paste full text, excerpts, or tweet text here..."
    )

    if analyze_clicked or (preset_title and selected_sample != "Custom Input (Type/Paste your own)"):
        if not input_title.strip() and not input_text.strip():
            st.warning("Please enter a headline or article text to analyze.")
        else:
            with st.spinner("Analyzing text semantics, linguistic features, and n-gram patterns..."):
                prediction = model_manager.predict(input_title, input_text, model_type=clean_model_name)
                heuristics = analyze_linguistic_features(input_title, input_text)
                
            verdict = prediction["verdict"]
            confidence = prediction["confidence"]
            fake_prob = prediction["fake_probability_percent"]
            real_prob = prediction["real_probability_percent"]
            
            st.markdown("---")
            
            # Primary Verdict Banner
            res_col1, res_col2 = st.columns([1.2, 1.8])
            
            with res_col1:
                if verdict == "FAKE":
                    st.markdown(f"""
                    <div class="verdict-fake">
                        <h2 style="color: #ef4444; margin: 0; font-size: 2.2rem;">🚨 UNRELIABLE / FAKE</h2>
                        <p style="margin: 8px 0 0 0; font-size: 1.15rem; font-weight: 600; color: #fca5a5;">
                            Confidence: {confidence:.1f}%
                        </p>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #fecaca;">
                            High probability of fabricated, misleading, or conspiratorial narrative.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-real">
                        <h2 style="color: #22c55e; margin: 0; font-size: 2.2rem;">✅ CREDIBLE / REAL</h2>
                        <p style="margin: 8px 0 0 0; font-size: 1.15rem; font-weight: 600; color: #86efac;">
                            Confidence: {confidence:.1f}%
                        </p>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #bbf7d0;">
                            Linguistic style matches verified reporting & institutional journalism.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # Probability distribution bar
                st.markdown("**Model Probability Gauge:**")
                st.progress(fake_prob / 100.0)
                p_c1, p_c2 = st.columns(2)
                p_c1.caption(f"Real: **{real_prob:.1f}%**")
                p_c2.caption(f"Fake: **{fake_prob:.1f}%**")
                
            with res_col2:
                # Stylistic & Linguistic Heuristics Breakdown
                st.markdown("#### **Stylistic & Clickbait Indicators**")
                
                h_col1, h_col2, h_col3 = st.columns(3)
                with h_col1:
                    st.metric(
                        "Sensationalism Index",
                        f"{heuristics['sensationalism_score']}/100",
                        delta=f"Risk: {heuristics['risk_level']}",
                        delta_color="inverse" if heuristics['risk_level'] == "High" else "normal"
                    )
                with h_col2:
                    st.metric("ALL-CAPS Ratio", f"{heuristics['caps_ratio_percent']}%", f"{heuristics['caps_word_count']} words")
                with h_col3:
                    st.metric("Exclamation Marks", heuristics['exclamation_count'], f"Excessive: {heuristics['excessive_punctuation_count']}")
                    
                # Clickbait triggers found
                if heuristics["flagged_clickbait_terms"]:
                    st.markdown("**Detected Deception / Clickbait Triggers:**")
                    triggers_html = "".join([f'<span class="badge-tag tag-danger">⚠️ {term}</span>' for term in heuristics["flagged_clickbait_terms"]])
                    st.markdown(triggers_html, unsafe_allow_html=True)
                else:
                    st.caption("✓ No overt sensationalist or clickbait trigger words flagged.")
                    
                if heuristics["credible_indicators_found"]:
                    st.markdown("**Institutional / Credible Phrasing Signals:**")
                    cred_html = "".join([f'<span class="badge-tag tag-success">✓ {term}</span>' for term in heuristics["credible_indicators_found"]])
                    st.markdown(cred_html, unsafe_allow_html=True)

            # Consensus & Explainability Section
            st.markdown("---")
            c_col1, c_col2 = st.columns([1, 1])
            
            with c_col1:
                st.markdown("#### **Multi-Model Consensus**")
                consensus_data = []
                for m_name, info in prediction["model_breakdown"].items():
                    consensus_data.append({
                        "Algorithm": m_name,
                        "Verdict": "🚨 FAKE" if info["verdict"] == "FAKE" else "✅ REAL",
                        "Fake Probability": f"{info['fake_probability']*100:.1f}%",
                        "Real Probability": f"{info['real_probability']*100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(consensus_data), use_container_width=True, hide_index=True)
                
            with c_col2:
                st.markdown("#### **N-Gram Explainability (Why this verdict?)**")
                st.caption("Top words/phrases in this text with highest learned mathematical weights:")
                
                exp = prediction["explanation"]
                if exp["fake_signals"] or exp["real_signals"]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    
                    terms = []
                    weights = []
                    colors = []
                    
                    for s in exp["fake_signals"][:4]:
                        terms.append(f"{s['term']} (Fake)")
                        weights.append(s["weight"])
                        colors.append("#ef4444")
                        
                    for s in exp["real_signals"][:4]:
                        terms.append(f"{s['term']} (Real)")
                        weights.append(-s["weight"])
                        colors.append("#22c55e")
                        
                    y_pos = np.arange(len(terms))
                    ax.barh(y_pos, weights, color=colors, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(terms, color='white', fontsize=9)
                    ax.tick_params(axis='x', colors='white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#374151')
                    ax.spines['bottom'].set_color('#374151')
                    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No dominant extreme TF-IDF vocabulary triggers matched.")

# =========================================================
# TAB 2: DATASET EXPLORER
# =========================================================
with tab_dataset:
    st.markdown("### Training Corpus & Data Analytics")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Labeled Articles", len(current_df))
    col_d2.metric("Real News Samples", int((current_df["label"] == "REAL").sum()))
    col_d3.metric("Fake News Samples", int((current_df["label"] == "FAKE").sum()))
    
    st.markdown("---")
    
    # Filter controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        label_filter = st.radio("Filter By Category:", ["ALL", "REAL", "FAKE"], horizontal=True)
    with f_col2:
        search_query = st.text_input("Search corpus by keyword:", placeholder="e.g. satellite, vaccine, microchip...")
        
    filtered_df = current_df.copy()
    if label_filter != "ALL":
        filtered_df = filtered_df[filtered_df["label"] == label_filter]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(search_query, case=False, na=False) |
            filtered_df["text"].str.contains(search_query, case=False, na=False)
        ]
        
    st.dataframe(
        filtered_df[["title", "category", "label", "text"]],
        use_container_width=True,
        hide_index=True,
        height=380
    )
    
    st.markdown("---")
    # Category Distribution Chart
    st.markdown("#### **Corpus Category Breakdown**")
    cat_counts = current_df.groupby(["category", "label"]).size().unstack(fill_value=0)
    
    fig_cat, ax_cat = plt.subplots(figsize=(8, 3.5))
    fig_cat.patch.set_facecolor('#0e1117')
    ax_cat.set_facecolor('#0e1117')
    cat_counts.plot(kind='bar', stacked=True, ax=ax_cat, color=['#ef4444', '#22c55e'])
    ax_cat.tick_params(colors='white')
    ax_cat.set_ylabel("Number of Articles", color='white')
    ax_cat.set_xlabel("Domain / Topic", color='white')
    ax_cat.legend(["Fake News", "Real News"], facecolor='#1f2937', edgecolor='#374151', labelcolor='white')
    ax_cat.spines['top'].set_visible(False)
    ax_cat.spines['right'].set_visible(False)
    ax_cat.spines['left'].set_color('#374151')
    ax_cat.spines['bottom'].set_color('#374151')
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig_cat)

# =========================================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# =========================================================
with tab_performance:
    st.markdown("### Model Benchmark Comparison")
    st.caption("Evaluated on stratified hold-out test split using Accuracy, Precision, Recall, and F1-Score:")
    
    perf_rows = []
    for model_name, m in model_manager.metrics.items():
        perf_rows.append({
            "Model Architecture": model_name,
            "Accuracy": f"{m['accuracy']*100:.1f}%",
            "Precision": f"{m['precision']*100:.1f}%",
            "Recall": f"{m['recall']*100:.1f}%",
            "F1-Score": f"{m['f1']:.3f}",
            "Test Set Size": m["test_samples"]
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### **Confusion Matrix Visualizer**")
    cm_model_choice = st.selectbox("Select model to inspect confusion matrix:", list(model_manager.metrics.keys()), index=0)
    
    cm = np.array(model_manager.metrics[cm_model_choice]["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
    fig_cm.patch.set_facecolor('#0e1117')
    ax_cm.set_facecolor('#0e1117')
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted REAL", "Predicted FAKE"],
        yticklabels=["Actual REAL", "Actual FAKE"],
        ax=ax_cm,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax_cm.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig_cm)
    
    st.markdown("---")
    st.markdown("#### **Retrain Models with Custom Split**")
    retrain_col1, retrain_col2 = st.columns([2, 1])
    with retrain_col1:
        test_size_slider = st.slider("Test Split Ratio:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    with retrain_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Retrain All Models", use_container_width=True):
            with st.spinner("Retraining NLP pipeline on dataset..."):
                model_manager.train_all(current_df, test_size=test_size_slider)
                st.success("Successfully retrained all models!")
                st.rerun()

# =========================================================
# TAB 4: UPLOAD CUSTOM DATASET & BENCHMARK GUIDE
# =========================================================
with tab_upload:
    st.markdown("### Upload Custom Dataset & Public Dataset Guide")
    st.markdown(
        "You can import any custom CSV file to train and benchmark models on your own news corpus."
    )
    
    st.markdown("#### **1. Recommended Datasets & Formats**")
    with st.expander("📖 What type of dataset is needed? (Click to view full specs)", expanded=True):
        st.markdown("""
        To train a fake news detector, a dataset needs **three core elements**:
        
        | Column Name | Recommended Name | Content Description |
        | :--- | :--- | :--- |
        | **Title / Headline** | `title` | The news article title or headline. |
        | **Article Text** | `text` or `content` | The full body or summary paragraph of the article. |
        | **Class Label** | `label` | `0` or `REAL` for reliable news; `1` or `FAKE` for fabricated news. |
        
        #### Top Public Academic Datasets:
        1. **WELFake Dataset (72,134 news articles)**
           - High quality, balanced, merges Kaggle, McIntire, Reuters, and BuzzFeed datasets.
           - Available on Kaggle & IEEE DataPort.
        2. **ISOT Fake News Dataset (University of Victoria)**
           - Over 44,000 articles (Real news from Reuters, Fake news from flagged propaganda sites).
        3. **LIAR Dataset (William Yang Wang, UCSB)**
           - 12.8k human-labeled short statements from PolitiFact with truth ratings.
        4. **FakeNewsNet (Penn State)**
           - Multi-dimensional dataset with social context, tweets, and fact-checking from PolitiFact & GossipCop.
        """)
        
    st.markdown("#### **2. Upload Your CSV File**")
    uploaded_file = st.file_uploader("Upload a CSV dataset:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV successfully! Found {len(custom_df)} rows and {len(custom_df.columns)} columns.")
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(custom_df.head(5), use_container_width=True)
            
            st.markdown("#### **Map Columns**")
            col_u1, col_u2, col_u3 = st.columns(3)
            with col_u1:
                title_col = st.selectbox("Headline/Title column:", custom_df.columns, index=0)
            with col_u2:
                # Default to text or second column
                default_text_idx = 1 if len(custom_df.columns) > 1 else 0
                for idx, c in enumerate(custom_df.columns):
                    if "text" in c.lower() or "content" in c.lower() or "body" in c.lower():
                        default_text_idx = idx
                        break
                text_col = st.selectbox("Text/Body column:", custom_df.columns, index=default_text_idx)
            with col_u3:
                default_label_idx = len(custom_df.columns) - 1
                for idx, c in enumerate(custom_df.columns):
                    if "label" in c.lower() or "target" in c.lower() or "class" in c.lower():
                        default_label_idx = idx
                        break
                label_col = st.selectbox("Label/Target column:", custom_df.columns, index=default_label_idx)
                
            if st.button("🚀 Train Veritas AI on Uploaded Dataset", type="primary"):
                with st.spinner("Processing text, vectorizing, and training models..."):
                    # Normalize columns
                    clean_custom_df = pd.DataFrame({
                        "title": custom_df[title_col].fillna(""),
                        "text": custom_df[text_col].fillna(""),
                        "label": custom_df[label_col].astype(str)
                    })
                    
                    # Save to local data path
                    clean_custom_df.to_csv(DATA_PATH, index=False)
                    st.session_state.data_version += 1
                    get_model_manager.clear()
                    st.success("Trained successfully on your custom dataset!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
