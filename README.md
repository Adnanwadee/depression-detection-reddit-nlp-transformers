<div align="center">

# Depression Detection from Reddit Text Using NLP and Transformers

### Binary text classification project for detecting depression-related Reddit posts using transformer-based NLP models

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-1F6FEB?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFCC4D?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MiniLM](https://img.shields.io/badge/MiniLM-Best%20Trade--off-2EA44F?style=for-the-badge)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Transformer%20Model-6F42C1?style=for-the-badge)
![ELECTRA](https://img.shields.io/badge/ELECTRA-Small%20Discriminator-0969DA?style=for-the-badge)

</div>

---

## Overview

This project develops an NLP-based machine learning system for detecting depression-related language in Reddit posts.

The task is formulated as a binary text classification problem. Given a Reddit post, the model predicts whether the text is depression-related or non-depression-related.

The project compares three transformer-based models:

- MiniLM
- DistilBERT
- ELECTRA-small

The workflow includes dataset preprocessing, class balancing, transformer tokenization, model fine-tuning, evaluation, interpretability using LIME, and an optional Gradio interface for local demonstration.

---

## Problem Statement

Social media platforms contain large volumes of user-generated text that may reflect emotional distress or mental health concerns. Reddit is especially useful for this kind of NLP task because users often write long-form posts describing personal experiences, thoughts, and emotional states.

The goal of this project is to classify Reddit text into two categories:

| Label | Meaning |
|---|---|
| `0` | Non-depressed |
| `1` | Depressed |

This project is intended for academic NLP experimentation. It is not a medical diagnosis system.

---

## Project Objectives

The main objectives are:

- Build a complete NLP pipeline for depression-related Reddit text classification.
- Preprocess raw Reddit posts and remove non-semantic shortcut features.
- Prevent label leakage by removing metadata that can expose the source subreddit.
- Balance the dataset using undersampling.
- Fine-tune multiple transformer models for binary sequence classification.
- Compare MiniLM, DistilBERT, and ELECTRA-small.
- Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
- Add interpretability using LIME to understand influential words.
- Provide a local GUI demonstration using Gradio.

---

## Dataset

The project uses the Kaggle Reddit Depression Dataset.

The original dataset contains approximately:

```text
2.47 million Reddit posts
```

### Original Features

| Feature | Description |
|---|---|
| `title` | Reddit post title |
| `body` | Reddit post body |
| `subreddit` | Source subreddit |
| `created_utc` | Post creation timestamp |
| `upvotes` | Number of upvotes |
| `num_comments` | Number of comments |
| `label` | Target label: 0 = non-depressed, 1 = depressed |

---

## Dataset Availability

The dataset is not included in this repository.

Expected local path used by the preprocessing notebook:

```text
Data/reddit_depression_dataset.csv
```

The processed balanced dataset expected by the model notebooks is:

```text
Data/balanced_reddit_data.csv
```

The data files are intentionally excluded to keep the repository lightweight and to avoid redistributing external dataset content.

---

## Data Preprocessing

The preprocessing pipeline is implemented in:

```text
notebooks/Preprocessing.ipynb
```

### Main Preprocessing Steps

| Step | Description |
|---|---|
| Load dataset | Reads the raw Reddit depression dataset |
| Drop metadata columns | Removes `created_utc`, `subreddit`, and `Unnamed: 0` |
| Remove missing rows | Drops rows with missing `title`, `body`, `upvotes`, `num_comments`, or `label` |
| Convert numeric fields | Converts `upvotes` and `num_comments` to integers |
| Merge text fields | Combines `title` and `body` into one `text` column |
| Clean text | Removes URLs, emojis, digits, punctuation, line breaks, and extra spaces |
| Lowercase text | Converts all text to lowercase |
| Save cleaned data | Saves `cleaned_reddit_data.csv` |
| Balance classes | Applies undersampling to the majority class |
| Save balanced data | Saves `balanced_reddit_data.csv` |

### Columns Removed

The following columns were removed:

```text
created_utc
subreddit
Unnamed: 0
```

The `subreddit` column was removed because it can create label leakage. For example, posts from depression-related communities can reveal the target label without requiring the model to understand the actual text.

---

## Text Cleaning

The text cleaning function performs the following operations:

```text
Remove URLs
Remove emojis and non-BMP Unicode characters
Remove line breaks and tabs
Remove digits
Remove non-alphabetic characters
Convert text to lowercase
Normalize extra spaces
```

After cleaning, each sample contains a single normalized text field.

---

## Dataset Size After Cleaning

After full cleaning, the dataset shape becomes:

```text
1,925,427 rows
4 columns
```

Remaining columns:

| Column | Description |
|---|---|
| `text` | Cleaned Reddit post text |
| `upvotes` | Number of upvotes |
| `num_comments` | Number of comments |
| `label` | Classification label |

---

## Class Distribution

Before undersampling, the cleaned dataset is imbalanced:

| Class | Count |
|---|---:|
| Non-depressed | 1,499,842 |
| Depressed | 425,585 |

To reduce majority-class bias, undersampling is applied to the non-depressed class.

After undersampling:

| Class | Count |
|---|---:|
| Non-depressed | 425,585 |
| Depressed | 425,585 |

Final balanced dataset size:

```text
851,170 rows
```

---

## Model Development Workflow

All transformer models follow the same general pipeline.

```text
Load balanced dataset
        |
        v
Split into train, validation, and test sets
        |
        v
Load model-specific tokenizer
        |
        v
Tokenize text in batches
        |
        v
Create custom PyTorch Dataset
        |
        v
Load pretrained transformer model
        |
        v
Configure Hugging Face Trainer
        |
        v
Fine-tune the model
        |
        v
Evaluate on test set
        |
        v
Generate classification report and confusion matrix
        |
        v
Save local model artifacts
```

---

## Train/Validation/Test Split

The balanced dataset is split as follows:

| Split | Percentage | Size |
|---|---:|---:|
| Training | 70% | 595,819 |
| Validation | 15% | 127,675 |
| Test | 15% | 127,676 |

The split uses stratification to preserve class balance.

---

## Models Implemented

### MiniLM

Notebook:

```text
notebooks/Mini LM Model.ipynb
```

Model used:

```text
nreimers/MiniLM-L6-H384-uncased
```

MiniLM is a compact transformer model designed for efficient inference while preserving strong language representation quality.

In this project, MiniLM provided the best practical balance between model size, performance, and deployment efficiency.

---

### DistilBERT

Notebook:

```text
notebooks/distilbert_base.ipynb
```

Model used:

```text
distilbert-base-uncased
```

DistilBERT is a compressed version of BERT. It keeps strong language understanding capability while reducing model size and improving inference speed.

In this project, DistilBERT achieved strong classification performance and matched MiniLM at the rounded metric level.

---

### ELECTRA-small

Notebook:

```text
notebooks/electra-small-discriminator.ipynb
```

Model used:

```text
google/electra-small-discriminator
```

ELECTRA-small uses a replaced-token detection pretraining objective and is designed to be computationally efficient.

In this project, ELECTRA-small achieved slightly lower overall performance than MiniLM and DistilBERT but remained a strong lightweight baseline.

---

## Training Configuration

The model notebooks use Hugging Face `Trainer` with a consistent training setup.

| Setting | Value |
|---|---|
| Learning rate | `2e-5` |
| Train batch size | `32` |
| Evaluation batch size | `32` |
| Epochs | `3` |
| Weight decay | `0.01` |
| Evaluation strategy | Per epoch |
| Save strategy | Per epoch |
| Best model metric | Accuracy |
| Early stopping | Enabled |

Tokenization is performed in batches to reduce memory pressure on large datasets.

---

## Evaluation Metrics

The models are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Measures total correct predictions |
| Precision | Measures reliability of positive predictions |
| Recall | Measures ability to detect depressed samples |
| F1-score | Balances precision and recall |
| Confusion matrix | Shows class-level prediction errors |

For this task, recall and F1-score are especially important because false negatives can be more sensitive in mental-health-related text classification.

---

## Model Results

### Summary Table

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| MiniLM | 0.95 | 0.95 | 0.95 | 0.95 |
| DistilBERT | 0.95 | 0.95 | 0.95 | 0.95 |
| ELECTRA-small | 0.94 | 0.94 | 0.94 | 0.94 |

### MiniLM Test Results

| Metric | Value |
|---|---:|
| Accuracy | 0.9468 |
| Precision | 0.9403 |
| Recall | 0.9543 |
| F1-score | 0.9472 |
| Test samples | 127,676 |

MiniLM showed strong recall on the depressed class and provided the best practical trade-off between performance and model efficiency.

### DistilBERT Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Not Depressed | 0.95 | 0.95 | 0.95 | 63,838 |
| Depressed | 0.95 | 0.95 | 0.95 | 63,838 |

Overall accuracy:

```text
0.95
```

### ELECTRA-small Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Not Depressed | 0.95 | 0.93 | 0.94 | 63,838 |
| Depressed | 0.93 | 0.96 | 0.94 | 63,838 |

Overall accuracy:

```text
0.94
```

---

## Comparative Analysis

| Model | Strength |
|---|---|
| MiniLM | Best balance between accuracy, model size, and deployment practicality |
| DistilBERT | Strong general classification performance |
| ELECTRA-small | Efficient and lightweight, with strong recall for depressed samples |

Final project conclusion:

```text
MiniLM is the most balanced model in this experiment.
```

It combines compact architecture, strong classification results, and practical suitability for limited-resource environments.

---

## Interpretability

The project applies LIME to interpret model predictions.

LIME is used to highlight the words that most influenced each classification decision. This helps verify whether the model is relying on meaningful linguistic patterns rather than irrelevant artifacts.

Implemented interpretability notebooks include LIME analysis for:

```text
MiniLM
DistilBERT
ELECTRA-small
```

Interpretability is important in this project because mental-health-related classification should not be treated as a black-box decision process.

---

## Gradio Demonstration

The project includes a Gradio notebook:

```text
notebooks/Depression Gradio (GUI).ipynb
```

The GUI allows a user to enter a Reddit-style sentence and receive side-by-side predictions from:

```text
DistilBERT
ELECTRA-small
MiniLM
```

Each model returns:

```text
Predicted label
Confidence percentage
```

Important note:

The GUI notebook requires locally saved model folders. These trained model weights are not included in this repository.

---

## Model Artifacts

Trained model weights are intentionally excluded from this repository.

Excluded artifacts include:

```text
models/
saved_model_*/
*.safetensors
*.bin
*.pt
*.pth
*.pkl
*.joblib
```

This repository is designed to show the methodology, training workflow, and evaluation process without publishing ready-to-use model weights.

To reproduce the models, run the preprocessing notebook and then train the model notebooks locally.

---

## Repository Structure

```text
depression-detection-reddit-nlp-transformers/
│
├── docs/
│   └── depression_detection_report.pdf
│
├── notebooks/
│   ├── Depression Gradio (GUI).ipynb
│   ├── Mini LM Model.ipynb
│   ├── Preprocessing.ipynb
│   ├── distilbert_base.ipynb
│   └── electra-small-discriminator.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Adnanwadee/depression-detection-reddit-nlp-transformers.git
```

### 2. Open the Project Folder

```bash
cd depression-detection-reddit-nlp-transformers
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset Locally

Create a local `Data` folder:

```text
Data/
└── reddit_depression_dataset.csv
```

The dataset is not included in this repository.

### 5. Run Preprocessing

Open and run:

```text
notebooks/Preprocessing.ipynb
```

This creates:

```text
Data/cleaned_reddit_data.csv
Data/balanced_reddit_data.csv
```

### 6. Train the Models

Run one or more model notebooks:

```text
notebooks/Mini LM Model.ipynb
notebooks/distilbert_base.ipynb
notebooks/electra-small-discriminator.ipynb
```

### 7. Run the GUI Demo

After training and saving the models locally, run:

```text
notebooks/Depression Gradio (GUI).ipynb
```

The GUI will not work unless the saved model folders exist locally.

---

## Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Notebook Environment | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Transformer Framework | Hugging Face Transformers |
| Training Interface | Hugging Face Trainer |
| Interpretability | LIME |
| GUI Demo | Gradio |
| Visualization | Matplotlib |

---

## Report

The academic report is included in:

```text
docs/depression_detection_report.pdf
```

The report covers:

- Problem background
- Dataset overview
- Preprocessing decisions
- Model development workflow
- Transformer model comparison
- Training and evaluation results
- Confusion matrices
- LIME interpretability
- Gradio demonstration

---

## Limitations

- The dataset is not included in the repository.
- Trained model weights are not included.
- The GUI requires local saved models before it can run.
- The project is an academic NLP experiment, not a clinical diagnostic system.
- The models classify text patterns and should not be used to diagnose depression.
- Results depend on dataset quality, labeling assumptions, and preprocessing decisions.
- Reddit posts may contain sarcasm, ambiguity, slang, or incomplete context.
- Undersampling balances the classes but removes many majority-class samples.
- The notebooks are experiment-oriented and are not structured as a production package.

---

## Ethical Considerations

This project deals with mental-health-related language. Any use of the models should be handled carefully.

Important constraints:

- The model output must not be interpreted as a medical diagnosis.
- Predictions should be treated as NLP classification results only.
- Human review is necessary in any real mental-health context.
- Dataset privacy and licensing must be respected.
- The system should not be used for surveillance, profiling, or automated judgment of individuals.
- False positives and false negatives may both have sensitive consequences.

---

## Future Improvements

Potential improvements include:

- Add classical ML baselines such as TF-IDF with Logistic Regression or SVM.
- Add ROC-AUC and PR-AUC evaluation.
- Add more detailed error analysis for false positives and false negatives.
- Add experiment tracking for model versions and metrics.
- Replace undersampling with more advanced imbalance-handling strategies.
- Add reproducible configuration files for training.
- Export a controlled inference script without publishing trained weights.
- Refactor notebooks into a clean Python package.
- Add model cards for each transformer model.
- Add stronger privacy and ethical documentation.
- Add cross-validation or external test-set validation.
- Add SHAP or other explainability methods for comparison with LIME.

---

## Authors

| Name |
|---|
|  Adnan Wadee Abdullah |
| Abdelrahman Abu Naser |
| Ammar Alrousan |
| Ra'ad Shraiedeh |

---

## Project Category

```text
Natural Language Processing
Text Classification
Transformer Models
Mental Health NLP
Depression Detection
Reddit Text Analysis
Deep Learning
Machine Learning
```

---

<div align="center">

### Final Summary

This repository presents a transformer-based NLP project for classifying Reddit posts as depression-related or non-depression-related.

The project demonstrates a complete academic machine learning workflow: preprocessing, balancing, transformer fine-tuning, evaluation, interpretability, and local GUI demonstration.

**Best practical model:** MiniLM  
**Top rounded performance:** 0.95 accuracy and F1-score

</div>
