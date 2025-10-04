# Applied Machine Learning Project

This is a collection of exercises for a ** Machine Learning** course. It includes examples of **polynomial regression**, **ridge regression**, **K‑Nearest Neighbors (KNN)** classification, and a full **NLP pipeline** with a custom implementation of the **Multinomial Naive Bayes** model.

## Directory Structure

```
ml1/
├─ 1.txt                     # Short theory answers (draft/TODO)
├─ 2a.ipynb                  # Polynomial regression on bottle.csv
├─ 2b.ipynb                  # Ridge regression on bottle.csv
├─ 3a.ipynb                  # KNN (2 features, k=3) on iris.csv
├─ 3b.ipynb                  # KNN (2 features, k=1..15) on iris.csv
├─ 3c.ipynb                  # KNN (4 features, k=1..15) on iris.csv
└─ 4.ipynb                   # NLP pipeline + Multinomial NB implementation
```

## Requirements

- Python 3.8+
- Jupyter Notebook
- Required libraries:
  - `numpy`, `pandas`, `matplotlib`
  - `nltk` (for 4.ipynb)

## Required Datasets

- `bottle.csv` (for 2a, 2b): contains `Salnty` and `T_degC` columns
- `iris.csv` (for 3a, 3b, 3c): contains sepal/petal features + `species`
- text CSV file (for 4): text content and label column

> All notebooks use `google.colab.files.upload()` for manual file upload.

## How to Run

1. Open notebook in Jupyter or Google Colab
2. Upload the required `.csv` file when prompted
3. Run the code cells in order (`Shift + Enter`)

## 🔍 Notebook Overview

### `1.txt`
- Draft answers in Serbian: Word2Vec, Naive Bayes, linear separability, etc.
- Can be expanded into exam-ready text.

### `2a.ipynb` — Polynomial Regression
- Dataset: `bottle.csv`
- Target: `T_degC`, Feature: `Salnty`
- Functions: `create_design_matrix`, `fit_polynomial_regression`, `predict`, `mean_squared_error`
- Plots regression curves and MSE for multiple degrees.

### `2b.ipynb` — Ridge Regression
- Same dataset and features as 2a.
- Uses `fit_ridge_regression(X, y, lambda)` with L2 regularization.
- Plots cost vs. lambda graph.

### `3a.ipynb` — KNN (k=3)
- Dataset: `iris.csv`
- Features: `sepal_length`, `sepal_width`
- Implements `predict_knn`, `euclidean_distance` manually.

### `3b.ipynb` — KNN with varying k (1–15)
- Same features as 3a.
- Tests multiple k values and plots accuracy.

### `3c.ipynb` — KNN with all 4 features
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Uses full feature set, shows improved classification accuracy.

### `4.ipynb` — NLP + Multinomial Naive Bayes
- Uses `nltk` for:
  - tokenization
  - POS tagging
  - lemmatization
  - stopwords removal
- Custom class: `MultinomialNaiveBayes` with `fit()` and `predict()`
- Full pipeline from raw CSV to text classification.
