# 💬 NLP Review Analytics

End-to-end Natural Language Processing project analyzing customer reviews using sentiment analysis, topic modeling, and machine learning classification.

---

## 🔍 What This Project Does

- **Sentiment Analysis** with VADER — rule-based NLP model scoring each review
- **Topic Modeling** with LDA — discovers 5 hidden topics across all reviews
- **ML Classifier** — Logistic Regression model predicting positive/negative sentiment
- **Word Clouds** — visual representation of most frequent words per sentiment
- **Power BI Dashboard** — interactive visualization of all NLP insights

---

## 💡 Key Insights

- VADER sentiment analysis achieves strong alignment with ground truth labels
- Positive reviews tend to use more specific, descriptive vocabulary
- Negative reviews are typically shorter but more emotionally charged
- Topic modeling reveals clear thematic clusters across product categories

---

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| Python 3.12 | Data processing and NLP pipeline |
| NLTK + VADER | Sentiment analysis and text preprocessing |
| scikit-learn | TF-IDF, LDA topic modeling, Logistic Regression |
| WordCloud | Visual word frequency analysis |
| pandas | Data manipulation |
| Power BI | Interactive dashboard |
| Git | Version control |

---

## 📁 Project Structure

nlp-review-analytics/
├── data/
│   ├── raw/                    ← generated review dataset
│   └── processed/              ← preprocessed data for Power BI
├── notebooks/
│   └── nlp_analysis.ipynb      ← full NLP analysis notebook
├── scripts/
│   ├── collect_data.py         ← synthetic data generation
│   └── preprocess.py           ← text cleaning and preprocessing
├── outputs/                    ← exported charts and word clouds
├── powerbi/
│   └── dashboard.pbix          ← Power BI dashboard
└── README.md

---

## ▶️ How to Run
```bash
git clone https://github.com/fertms/nlp-review-analytics.git
cd nlp-review-analytics

python -m venv .venv
.venv\Scripts\activate

pip install pandas numpy matplotlib seaborn nltk textblob wordcloud scikit-learn ipykernel

cd scripts
python collect_data.py
python preprocess.py

cd ../notebooks
jupyter notebook nlp_analysis.ipynb
```

---

## 📊 Models Used

| Model | Type | Purpose |
|---|---|---|
| VADER | Rule-based NLP | Sentiment scoring |
| LDA | Unsupervised ML | Topic discovery |
| Logistic Regression | Supervised ML | Sentiment classification |
| TF-IDF | Feature extraction | Text vectorization |