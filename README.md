# Loan Default Prediction — Neural Network Classifier

LendingClub loan data contains ~396,000 records of real borrowers. The goal is to predict whether a loan will be fully repaid or charged off. The problem encountered in this dataset is class imbalance. The split is 20% to 80% in favour of paid off loans, so without proper implementation the model will predict paid off almost every time, which leads to a broken model. This project focuses on handling that imbalance correctly.

---

## Dataset

- **Source:** LendingClub via Kaggle (`lending_club_loan_two.csv`)
- **Size:** ~396,000 loans
- **Target:** `loan_status` — Fully Paid (1) or Charged Off (0)
- **Features:** 27 original columns covering loan amount, interest rate, borrower employment, credit history, and more

---

## What This Project Covers

- Exploratory data analysis — column selection based on cardinality, missing values, and leakage risk
- Feature engineering — month and year extracted from `earliest_cr_line` before dropping the original column
- Full sklearn preprocessing pipeline using `ColumnTransformer` — median imputation and MinMax scaling for numerical columns, most frequent imputation and one-hot encoding for categorical columns
- Stratified train/test split to preserve class ratio
- Class imbalance handled via `compute_class_weight` — no oversampling
- Neural network: `64 → 32 → 1` architecture with dropout and early stopping

---

## How to Run

```
loan-default-prediction/
│
├── DATA/
│   └── lending_club_loan_two.csv
├── lending_club_loan.ipynb
└── README.md
```

The dataset is not included in this repository due to file size. Download it from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place it in the `DATA/` folder.

Launch the notebook:

```bash
jupyter notebook lending_club_loan.ipynb
```

Run all cells top to bottom.

---

## Results

The model achieves 67% overall accuracy and 63% recall on the default class (class 0 in this project). Previous iterations of the model were reaching ~80% accuracy, while the default class was almost never picked. The drop in accuracy made overall scores lower, but the model itself became reliable.

---

## Tech Stack

Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras
