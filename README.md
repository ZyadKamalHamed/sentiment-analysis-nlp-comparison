# Sentiment Analysis Comparison: NLP Models for App Review Classification

Comparing three classical NLP models (Naive Bayes, SVM, Logistic Regression) for binary sentiment classification on ChatGPT app reviews.

## Overview

This project investigates how different machine learning classifiers perform sentiment analysis on 8,000 Google Play Store reviews. Beyond just finding the most accurate model, I wanted to understand how each approach handles ambiguous neutral sentiment differently.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **91.1%** | 91.2% | 91.2% | 91.2% |
| Naive Bayes | 88.8% | 89.1% | 88.8% | 88.7% |
| Logistic Regression | 84.0% | 84.0% | 84.0% | 84.0% |

SVM performed best with balanced precision and recall across both classes.

## Dataset

- **Source:** Google Play Store reviews (ChatGPT app)
- **Size:** 8,000 reviews (4,000 positive / 4,000 negative)
- **Split:** 80/20 train-test (6,400 train / 1,600 test)
- **Labels:** Binary (1-star = negative, 5-star = positive)

## Implementation

### Preprocessing Pipeline
```python
# Text cleaning
- Lowercase conversion
- Punctuation removal  
- Whitespace normalisation
- Stop word removal (NLTK)
- Tokenisation
```

### Feature Extraction
TF-IDF Vectorisation converts text to numerical features representing word importance.

### Models

**1. Support Vector Machine (LinearSVC)**
Finds optimal hyperplane in high-dimensional space and maximises margin between classes. Works well with linearly separable text data.

**2. Naive Bayes Classifier**
Probabilistic approach using Bayes' theorem. Assumes feature independence, trains fast, provides good baseline.

**3. Logistic Regression**
Linear model with probability-based predictions. Interpretable feature weights with balanced performance.

## Interesting Finding

When tested on 4,000 ambiguous neutral reviews (3-star):
- **LR & SVM:** Classified 56-64% as negative (conservative bias)
- **Naive Bayes:** Classified 70% as positive (optimistic bias)

High test accuracy doesn't guarantee nuanced real-world interpretation.

## Tech Stack

- **Python 3.x**
- **Environment:** Google Colab
- **Libraries:** scikit-learn, NLTK, pandas, numpy, matplotlib, google-play-scraper

## Usage
```bash
# Clone repo
git clone https://github.com/ZyadKamalHamed/sentiment-analysis-nlp-comparison.git

# Open in Google Colab
# Upload 40-25561729-notebook.ipynb
# Run all cells
```

## Project Structure
```
sentiment-analysis-nlp-comparison/
├── 40-25561729-notebook.ipynb    # Main notebook
├── README.md                       # Documentation
└── requirements.txt                # Dependencies
```

## Context

**Course:** Introduction to Artificial Intelligence Application  
**Institution:** UTS  
**Year:** 2025  

## Potential Improvements

- Test transformer models (BERT, RoBERTa)
- Expand to multi-class (1-5 stars)
- Compare across different LLM apps
- Track sentiment trends over app versions

## Contact

**Zyad Kamal Hamed**  
- LinkedIn: [linkedin.com/in/zyadkamalhamed](https://linkedin.com/in/zyadkamalhamed/)
- Email: zyad2408@live.com.au
