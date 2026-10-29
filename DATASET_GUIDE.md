# 📚 Dataset Guide: Fake News Detection

This guide explains **what type of dataset is needed**, how to structure it, and where to find the **best public benchmark datasets** for training production-grade fake news classifiers.

---

## 1. What Type of Dataset Do You Need?

To train machine learning models (TF-IDF, Passive-Aggressive, Naive Bayes, Transformers) for fake news detection, your dataset should be tabular (CSV or JSON) containing labeled examples of both **credible** and **unreliable/fabricated** journalism.

### Recommended CSV Schema

| Column Name | Data Type | Required? | Description & Examples |
| :--- | :--- | :--- | :--- |
| **`title`** | String | **Yes** | The article's headline. *(e.g., "NASA James Webb Telescope Detects Water Vapor in Exoplanet")* |
| **`text`** | String | **Yes** | The full body or summary paragraph of the article. |
| **`label`** | String / Int | **Yes** | Target ground-truth class:<br>• `REAL` or `0` (Credible / Fact-checked genuine news)<br>• `FAKE` or `1` (Fabricated / Deceptive / Clickbait) |
| `category` | String | *Optional* | Subject domain: *Politics, Technology, Health, Science, Economy* |
| `source` | String | *Optional* | News agency or domain name where published |

---

## 2. Top Public Academic Datasets (Ready to Download)

If you want large-scale datasets (tens of thousands of articles), these are the gold-standard academic benchmarks:

### 1. WELFake Dataset (72,134 Articles) — *Highly Recommended*
- **Description:** One of the most comprehensive and balanced datasets available. It merges four popular datasets (Kaggle, McIntire, Reuters, and BuzzFeed) to prevent topic overfitting.
- **Size:** 72,134 articles (35,028 Real, 37,106 Fake).
- **Format:** CSV with `title`, `text`, `label`.
- **Where to get it:**
  - [Kaggle WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
  - [IEEE DataPort WELFake](https://ieee-dataport.org/open-access/welfake-word-embedding-over-linguistic-features-fake-news-detection)

### 2. ISOT Fake News Dataset (University of Victoria)
- **Description:** Created by the Information Security and Object Technology (ISOT) Research Lab. Contains real-world news collected between 2016 and 2017. Real articles were scraped from Reuters.com, and fake articles were collected from unreliable websites flagged by PolitiFact.
- **Size:** 44,898 articles (21,417 Real, 23,481 Fake).
- **Format:** Two CSV files (`True.csv` and `Fake.csv`).
- **Where to get it:**
  - [University of Victoria ISOT Repository](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php)
  - [Kaggle ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 3. LIAR Benchmark Dataset (UCSB)
- **Description:** Benchmark dataset collected by William Yang Wang (UC Santa Barbara) from PolitiFact. Unlike binary datasets, LIAR features a 6-tier truth rating: *pants-on-fire, false, barely-true, half-true, mostly-true, true*.
- **Size:** 12,836 short human-labeled statements.
- **Where to get it:**
  - [Papers With Code LIAR Benchmark](https://paperswithcode.com/dataset/liar)
  - [Hugging Face `liar` Dataset](https://huggingface.co/datasets/liar)

### 4. FakeNewsNet (ASU / Penn State)
- **Description:** Multi-modal dataset featuring news from PolitiFact and GossipCop, accompanied by social media engagement (retweets, replies, user profiles) and fact-check editorial justifications.
- **Where to get it:**
  - [GitHub Repo: KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

---

## 3. Best Practices & Common Pitfalls

### ⚠️ Pitfall 1: Source Attribution Leakage
If every real news article starts with `"WASHINGTON (Reuters) -"` or `"LONDON (AP) -"`, the machine learning model will simply learn that the word `"Reuters"` means real news, rather than analyzing actual linguistic or factual cues.
*Solution:* Our preprocessing pipeline strips source prefixes and standard wire headers before vectorization.

### ⚠️ Pitfall 2: Class Imbalance
Having 95% real news and 5% fake news will cause models to achieve 95% accuracy simply by predicting "REAL" every time.
*Solution:* Always keep a balanced ratio (around 50% Real and 50% Fake), or apply class weights (`class_weight='balanced'`).

### ⚠️ Pitfall 3: Single-Domain Overfitting
If all fake news in your dataset is about political elections, the model will classify any article mentioning politics as "Fake".
*Solution:* Include diverse categories: Health, Technology, Science, Finance, Entertainment, and World Affairs.

---

## 4. How to Use Custom Datasets in This Prototype

1. Launch the web app (`streamlit run app.py`).
2. Navigate to **Tab 4: 📁 Upload Custom Dataset**.
3. Drag & drop your `.csv` file.
4. Select which columns correspond to **Title**, **Body Text**, and **Label**.
5. Click **"Train Veritas AI on Uploaded Dataset"** — the model will vectorize your data, train the ensemble, and update all benchmark graphs automatically.
