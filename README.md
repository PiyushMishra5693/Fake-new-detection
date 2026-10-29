# 🛡️ Veritas AI — Fake News Detection & Credibility Platform

An end-to-end Python Machine Learning application and interactive prototype for detecting deceptive news, conspiracy theories, propaganda, and clickbait articles.

---

## 🌟 Key Features

1. **Multi-Model Machine Learning Engine**:
   - **Passive-Aggressive Classifier**: Industry-standard high-speed online linear classifier for text streams.
   - **Logistic Regression**: Calibrated probabilistic estimates with L2 regularization.
   - **Multinomial Naive Bayes**: Frequency-based probabilistic baseline.
   - **Linear Support Vector Machine (LinearSVC)**: Robust margin-maximizing boundary classifier.
   - **Ensemble Consensus**: Aggregated voting and confidence score across all models.

2. **Stylistic & Clickbait Heuristic Engine**:
   - Sensationalism Risk Meter (0 to 100).
   - ALL-CAPS word frequency and ratio analysis.
   - Punctuation abuse tracker (exclamation & question mark spam).
   - Real-time flag for sensationalist trigger phrases (e.g., *"bombshell"*, *"suppressed miracle"*, *"what they don't want you to know"*, *"big pharma panicking"*).
   - Institutional and credible phrase indicators.

3. **N-Gram Model Explainability**:
   - Mathematical decomposition of active vocabulary terms.
   - Visual bar chart showing exactly which words or bigrams pushed the decision toward **FAKE** or **REAL**.

4. **Rich Interactive Web Interface (Streamlit)**:
   - **Live News Detector**: Paste any headline/body or select preloaded test articles with one click.
   - **Dataset Explorer**: Searchable corpus with category breakdowns and interactive charts.
   - **Model Benchmarks**: Live confusion matrix, Accuracy, Precision, Recall, and F1-Scores.
   - **Upload Custom Dataset**: Drag-and-drop CSV uploader with dynamic column mapper for retraining.

5. **Standalone CLI Mode**:
   - Rapid terminal verification: `python cli.py "Breaking news..."`.

---

## 🚀 Quick Start

### 1. Launch the Interactive Web Application
```bash
streamlit run app.py
```
This opens the web interface in your default browser at `http://localhost:8501`.

### 2. Run via Terminal / CLI
```bash
# Analyze a single headline:
python cli.py "BOMBSHELL: Secret 5G Satellite Transmits Mind Control Waves Through Tap Water!"

# Interactive terminal mode:
python cli.py --interactive
```

---

## 📁 Project Architecture

```
├── app.py                  # Streamlit web application interface
├── cli.py                  # Command-line testing utility
├── generate_dataset.py     # Balanced starter dataset builder
├── DATASET_GUIDE.md        # Comprehensive dataset specifications & public resources
├── requirements.txt        # Python package dependencies
├── data/
│   ├── news_dataset.csv    # Preloaded balanced dataset (Politics, Science, Tech, Health, Economy)
│   └── sample_articles.json# Ready-to-test verified sample articles
└── src/
    ├── __init__.py
    ├── preprocessor.py     # Text cleaning, normalization, and n-gram extraction
    ├── heuristics.py       # Stylistic, sensationalism, and clickbait detector
    └── model.py            # Model training, vectorization, calibration, & explainability
```

---

## 📚 Datasets

A balanced, multi-domain seed dataset is pre-generated in `data/news_dataset.csv`.
For large-scale production training (e.g. WELFake 72,000 articles or ISOT 44,000 articles), see [`DATASET_GUIDE.md`](DATASET_GUIDE.md).
