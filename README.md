# AI or Human? A Machine Learning Approach to Text Classification for Statistical Courses
With the increasing use of AI tools like ChatGPT in academia, distinguishing between human- and AI-generated responses is essential for maintaining academic integrity. This project explores a machine learning pipeline with Logistic Regression, SVM, MLP, and BERT models to classify text as human- or AI-generated based on linguistic and semantic features.

*This is a course project for STT 811 Applied Statistical Modeling for Data Scientists at MSU. The contributors are Mahnoor Sheikh, Andrew John J, Roshni Bhowmik and Ab Basit Syed Rafi.

🌐 Access the [streamlit web app](https://nlp-approach-to-ai-text-classification-stt-811.streamlit.app/) to delve into the detailed steps of data cleaning, preprocessing, and modelling, as well as to uncover the insights derived from the analysis.

## Table of Contents
1. [Dataset](#dataset)
2. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Modeling](#modeling)
5. [Streamlit App Features](#streamlit-app-features)
6. [Key Takeaways](#key-takeaways)
7. [References](#references)
8. [Installation and Usage](#installation-and-usage)

## Dataset
- **Source**: Custom dataset of 2,239 rows (from Mendeley)
- **Contents**:
  - `Question`: The original statistics question
  - `Human Response`: Text response from a student
  - `AI Response`: Text generated using a language model
- **Post-cleaning**: 1,993 usable examples

## Preprocessing and Feature Engineering
- **Cleaning**: Lowercasing, punctuation removal, tokenization, stopword removal
- **Feature Creation**:
  - Text length, special character counts
  - Flesch Reading Ease, Gunning Fog Index
  - Cosine similarity to question
  - Sentiment scores and sentiment gaps
- **Vectorization**: `CountVectorizer` followed by PCA (95% variance retained in 482 components)

## Exploratory Data Analysis
Key visuals and insights:
- **Top Trigrams** and **Common Words** in AI vs. Human responses
- **Word Clouds** and **Text Length Distribution**
- **Sentiment Gap Analysis** and **KDE Estimation**
- **Readability Scores**: AI responses are longer and more formulaic
- **Text Similarity**: AI more aligned with original questions
- **Pairplots & Correlation Heatmaps** reveal subtle response patterns

## Modeling
### Traditional ML Models
- Logistic Regression, Linear SVM, Decision Tree, Random Forest, KNN, Gradient Boosting, MLP
- **Best Accuracy**: ~85% (Logistic Regression, SVM, MLP)

### Deep Learning: BERT
- **Model**: `bert-base-uncased` via Hugging Face
- **Training**:
  - Tokenization (WordPiece)
  - 30 epochs with cross-entropy loss
  - AdamW optimizer
- **Performance**: Comparable to traditional models with potential for further gains

## Streamlit App Features
- Upload new questions and responses
- Evaluate text using trained models
- Visual analytics: word clouds, trigrams, readability, sentiment
- Compare AI vs. human characteristics interactively

## Key Takeaways
- Human responses were simpler, less verbose, and showed more variability
- AI responses were longer, sentimentally aligned with questions, and structurally consistent
- Readability, sentiment gap, and cosine similarity are strong distinguishing features
- The system offers a foundational step toward detecting AI-generated content in education

## References
- [Dataset on Mendeley](https://data.mendeley.com/datasets/mh892rksk2/4)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Installation and Usage
```bash
# Clone repo
git clone https://github.com/andrew-jxhn/STT811_StatsProject.git
cd STT811_StatsProject

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_code.py
```
