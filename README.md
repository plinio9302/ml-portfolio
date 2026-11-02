# Machine Learning Portfolio

**Author:** Plinio Durango
**Tools:** Python · scikit-learn · statsmodels · NLTK · pandas · matplotlib · seaborn · PyTorch · NumPy · lifelines · Jupyter

A collection of end-to-end machine learning projects covering regression, classification, ensemble methods, NLP, deep learning, forecasting, and simulation — built as part of coursework and independent study.

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
### 4. Mashable News Popularity — Tree-Based Methods
> Classification · Decision Tree · Bagging · Random Forest · Ensemble Methods · GridSearchCV

Predicts whether a Mashable news article will become popular (above-median share count) using 58 article features. Progressively improves from a single Decision Tree to Bagging to Random Forest, demonstrating the bias-variance tradeoff and the power of ensemble methods. Includes a companion spam email classifier built on the same ensemble pipeline with TF-IDF features.

**Repo:** [mashable_data](https://github.com/plinio9302/mashable_data) · **Notebooks:** [classification_tree.ipynb](https://github.com/plinio9302/mashable_data/blob/main/classification_tree.ipynb) · [Spam Classifier.ipynb](https://github.com/plinio9302/mashable_data/blob/main/Spam%20Classifier.ipynb)
**Key skills:** Decision Trees, Bagging, Random Forest, TF-IDF vectorization, GridSearchCV, cross-validation, feature importance, MSPE comparison

---


### 5. Neural Networks from Scratch
> Deep Learning · Autograd · CNN (Convolutional Neural Network) · Transformer · GCN (Graph Convolutional Network) · PyTorch

An end to end notebook in four parts that builds a scalar autograd engine with an MLP (Multi Layer Perceptron), a CNN, a Transformer, and a GCN from first principles. Each part is trained and then used to diagnose a real failure mode: a recursion limit in the backward pass, a double softmax that puts a floor of about 1.461 under the loss, a missing positional embedding, and a graph model that ignores its edges.

**Repo:** [neural_networks_from_scracth](https://github.com/plinio9302/neural_networks_from_scracth)
**Key skills:** Backpropagation, gradient checking, overfit one batch debugging, self attention, ablation studies, PyTorch

---

### 6. Installed Base Reliability and Service Cost Forecasting
> Survival Analysis · Weibull · Cox PH (Proportional Hazards) · Monte Carlo Simulation · Hierarchical Bayesian Models

A four part analysis of how a fleet of industrial machines fails and what servicing it will cost, framed around a semiconductor equipment maker pricing fixed price service contracts. Covers failure data cleaning, Weibull and Cox survival models, a Monte Carlo renewal simulation for quarterly cost (13.5% WAPE (Weighted Absolute Percentage Error) in backtests, against 9.8% for a naive baseline, reported as is), and a hierarchical Gamma Poisson model that gives the lowest error in the first 30 days of a new platform. Includes a leakage check that removed false telemetry drivers.

**Repo:** [Service_cost_forecasting_portafolio](https://github.com/plinio9302/Service_cost_forecasting_portafolio)
**Key skills:** Survival analysis, censoring, Weibull fitting, Monte Carlo simulation, Bayesian shrinkage, rolling origin backtesting, data leakage detection

---

### 7. World Cup Champion Prediction
> Elo Ratings · Poisson Model · Monte Carlo Simulation · NumPy

Estimates every national team's chance of winning the World Cup by simulating the full tournament 20,000 times, built from scratch with NumPy and pandas. Three layers: Elo ratings fitted on historical results, a Poisson match model that turns rating gaps into scorelines, and a Monte Carlo tournament simulation with FIFA tiebreak rules and penalty shootouts.

**Repo:** [worldcup-monte-carlo](https://github.com/plinio9302/worldcup-monte-carlo)
**Key skills:** Probabilistic modeling, simulation, Elo rating systems, Poisson distributions, uncertainty quantification

---

### 8. Breast Cancer Recurrence Prediction
> Classification · Class Imbalance · Random Forest · Cross-Validation · Recall

Predicts whether breast cancer will recur from 9 clinical features of 277 patients (UCI dataset). Compares five classifiers with stratified 5 fold cross validation and random oversampling, selects the Decision Tree on recall (0.872 in cross validation) because a missed recurrence is the costliest error, and ranks features by Random Forest importance, with tumor size first. Limitations of the small, older dataset are documented.

**Repo:** [breast-cancer-recurrence-prediction](https://github.com/plinio9302/breast-cancer-recurrence-prediction)
**Key skills:** Imbalanced classification, recall and ROC-AUC (Receiver Operating Characteristic, Area Under the Curve) evaluation, resampling, feature importance

---

## Repository Structure

```
ml-portfolio/
├── notebooks/          # Jupyter notebooks per project
├── data/               # Raw and processed datasets
├── reports/            # Reports and analyses
└── README.md
```

> **External repos:** [Housing_Case_Study](https://github.com/plinio9302/Housing_Case_Study) · [Decision_trees_classifier](https://github.com/plinio9302/Decision_trees_classifier) · [mashable_data](https://github.com/plinio9302/mashable_data) · [neural_networks_from_scracth](https://github.com/plinio9302/neural_networks_from_scracth) · [Service_cost_forecasting_portafolio](https://github.com/plinio9302/Service_cost_forecasting_portafolio) · [worldcup-monte-carlo](https://github.com/plinio9302/worldcup-monte-carlo) · [breast-cancer-recurrence-prediction](https://github.com/plinio9302/breast-cancer-recurrence-prediction)

## Skills Demonstrated

| Area | Technologies |
|------|-------------|
| Supervised Learning | Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Bagging |
| Model Selection | GridSearchCV, cross-validation, train/test split |
| Feature Engineering | TF-IDF, VIF, StandardScaler, encoding |
| NLP | N-gram models, text preprocessing, smoothing |
| Deep Learning | Autograd, CNN, Transformer, GCN, PyTorch |
| Survival and Forecasting | Kaplan Meier, Weibull, Cox PH, renewal simulation, hierarchical Bayesian models |
| Simulation | Monte Carlo, Elo ratings, Poisson models |
| Imbalanced Classification | Class weights, oversampling, recall, ROC-AUC |
| Libraries | scikit-learn, statsmodels, pandas, numpy, matplotlib, seaborn |

## How to Run

```bash
git clone https://github.com/plinio9302/ml-portfolio.git
cd ml-portfolio
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nltk torch scipy lifelines
jupyter notebook
```

---

*For SQL/data analysis projects, see [data-analysis-portfolio](https://github.com/plinio9302/data-analysis-portfolio)*
