# Indonesian Sentiment Analysis - SVC (PRDECT-ID)

Implementation of Support Vector Classification (SVC) for sentiment analysis on Indonesian product reviews using the PRDECT-ID dataset.

## Dataset

**PRDECT-ID Dataset** — Indonesian Product Reviews Dataset for Emotion Classification Tasks
- 5,400 product reviews from Tokopedia
- Labels: `Positive` / `Negative`
- Source: [Mendeley Data](https://data.mendeley.com/datasets/574v66hf2v/1)

**Dataset License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**
This dataset is licensed under the Creative Commons Attribution 4.0 International License. You are free to share, copy, and modify this dataset as long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made.

## Pipeline

1. Load dataset
2. Exploratory Data Analysis (class distribution & missing values)
3. Text cleaning (lowercase, remove URLs, symbols, numbers)
4. Stopword removal + Stemming (PySastrawi)
5. TF-IDF Vectorization (`max_features=3000`)
6. Train/test split (80% / 20%)
7. SVC training (`kernel=rbf`, `C=10`, `gamma=1`)
8. Evaluation (accuracy, classification report, confusion matrix)

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 91.3% |
| Precision (Negative) | 0.90 |
| Precision (Positive) | 0.92 |
| Recall (Negative) | 0.93 |
| Recall (Positive) | 0.89 |
| F1-Score (avg) | 0.91 |

## Requirements

```bash
pip install PySastrawi scikit-learn pandas numpy matplotlib seaborn
```

## Usage

1. Clone this repository
2. Upload `PRDECT-ID Dataset.csv` to your Google Drive
3. Adjust the dataset path in Cell 5 to match your Drive location
4. Run all cells in order

## References

- Jocelyne Dumlao. (2022). *PRDECT-ID: Indonesian Product Reviews Dataset for Emotion Classification Tasks*. Mendeley Data. https://doi.org/10.17632/574v66hf2v.1
- Dataset license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
