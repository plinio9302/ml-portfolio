# Machine Learning Portfolio

**Author:** Plinio Durango
**Tools:** Python · scikit-learn · statsmodels · NLTK · pandas · matplotlib · seaborn · Jupyter

A collection of end-to-end machine learning projects covering regression, classification, ensemble methods, and NLP — built as part of coursework and independent study.

---

## Projects

### 1. Housing Price Prediction — Ames Dataset
> Regression · Ridge · Lasso · LassoCV · VIF · Cross-Validation

Predicts residential sale prices using regularized regression on 80+ features. Full pipeline: EDA, feature engineering, multicollinearity detection (VIF), StandardScaler normalization, and comparative model evaluation (OLS vs. Ridge vs. Lasso).

**Repo:** [Housing_Case_Study](https://github.com/plinio9302/Housing_Case_Study)
**Key skills:** Ridge/Lasso regularization, hyperparameter tuning, statsmodels, feature importance

---

### 2. Spam Detection — Ensemble Methods
> Classification · Decision Tree · Bagging · Random Forest · TF-IDF · GridSearchCV

Classifies real .eml email files as spam or ham using tree-based models. Compares a single Decision Tree baseline against Bagging and Random Forest ensembles. Uses TF-IDF vectorization (5,000 features) and GridSearchCV for tuning.

**Repo:** [Decision_trees_classifier](https://github.com/plinio9302/Decision_trees_classifier) · **Notebook:** [spam_classifier.ipynb](https://github.com/plinio9302/ml-portfolio/blob/main/notebooks/spam_classifier.ipynb)
**Key skills:** Ensemble methods, NLP feature extraction, cross-validation, MSPE comparison

---

### 3. N-Gram Text Prediction
> NLP · Language Models · N-grams · Smoothing

Builds an N-gram language model for next-word prediction. Covers text preprocessing, N-gram construction, smoothing techniques, and predictive performance evaluation.

**Location:** `notebooks/` in this repository
**Key skills:** Tokenization, N-gram modeling, language model evaluation

---

## Repository Structure

```
ml-portfolio/
├── notebooks/          # Jupyter notebooks per project
├── data/               # Raw and processed datasets
├── reports/            # Reports and analyses
└── README.md
```

## Skills Demonstrated

| Area | Technologies |
|------|-------------|
| Supervised Learning | Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Bagging |
| Model Selection | GridSearchCV, cross-validation, train/test split |
| Feature Engineering | TF-IDF, VIF, StandardScaler, encoding |
| NLP | N-gram models, text preprocessing, smoothing |
| Libraries | scikit-learn, statsmodels, pandas, numpy, matplotlib, seaborn |

## How to Run

```bash
git clone https://github.com/plinio9302/ml-portfolio.git
cd ml-portfolio
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nltk
jupyter notebook
```

---

*For SQL/data analysis projects, see [data-analysis-portfolio](https://github.com/plinio9302/data-analysis-portfolio)*
