<div align="center">

# Depression Detection from Reddit Text Using NLP and Transformer Models

### Binary text classification for detecting depression-related Reddit posts using transformer-based NLP models

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-1F6FEB?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFCC4D?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MiniLM](https://img.shields.io/badge/MiniLM-Best%20Trade--off-2EA44F?style=for-the-badge)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Transformer%20Baseline-6F42C1?style=for-the-badge)
![ELECTRA](https://img.shields.io/badge/ELECTRA-Small%20Discriminator-0969DA?style=for-the-badge)

</div>

---

## Overview

This project presents an NLP-based depression detection system using Reddit text data and transformer-based language models.

The objective is to classify Reddit posts into two categories:

| Label | Meaning |
|---|---|
| `0` | Non-depressed |
| `1` | Depressed |

The project focuses on text preprocessing, class balancing, transformer fine-tuning, model comparison, interpretability, and a local Gradio-based demonstration interface.

This repository is intended as an academic machine learning project. It is not a clinical diagnostic tool.

---

## Problem Statement

Reddit contains large volumes of user-generated text where people may express emotions, distress, or mental health-related concerns. Natural Language Processing can be used to analyze these textual patterns and classify whether a post is likely to contain depression-related language.

The main challenge is to train models that learn from the linguistic content of the post rather than relying on shortcut metadata such as the source subreddit.

For that reason, the project removes metadata that may cause label leakage and trains transformer models on cleaned textual content.

---

## Project Objectives

The project aims to:

- Build a complete NLP pipeline for Reddit depression text classification.
- Preprocess large-scale Reddit text data.
- Remove metadata-based shortcuts that could cause label leakage.
- Merge post title and body into a single text representation.
- Balance the dataset using undersampling.
- Fine-tune multiple transformer-based models.
- Compare MiniLM, DistilBERT, and ELECTRA-small.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Add interpretability using LIME.
- Provide a local Gradio interface for model demonstration.

---

## Dataset

The project uses the Kaggle Reddit Depression Dataset.

The original dataset contains approximately:

```text
2.47 million Reddit posts
```

### Original Dataset Features

| Feature | Description |
|---|---|
| `title` | Reddit post title |
| `body` | Reddit post body |
| `subreddit` | Source subreddit |
| `created_utc` | Post creation timestamp |
| `upvotes` | Number of upvotes |
| `num_comments` | Number of comments |
| `label` | Target label: `0 = non-depressed`, `1 = depressed` |

---

## Dataset Availability

The dataset is not included in this repository.

This is intentional because:

- The dataset is large.
- The repository should remain lightweight.
- External dataset redistribution may depend on dataset licensing.
- The focus of this repository is the methodology, notebooks, report, and model comparison.

Expected local dataset paths may differ depending on whether the notebooks are executed locally or on Google Colab.

Recommended local structure:

```text
Data/
├── reddit_depression_dataset.csv
├── cleaned_reddit_data.csv
└── balanced_reddit_data.csv
```

If the notebooks are executed in Google Colab, update the dataset paths according to the uploaded file location.

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

## Preprocessing Pipeline

The preprocessing workflow is implemented in:

```text
notebooks/Preprocessing.ipynb
```

The preprocessing stage prepares the raw Reddit dataset for transformer-based training.

### Main Preprocessing Steps

| Step | Description |
|---|---|
| Load raw dataset | Reads the Reddit depression dataset |
| Remove unnecessary columns | Drops metadata and leakage-prone fields |
| Handle missing values | Removes rows with missing required values |
| Normalize numeric fields | Converts engagement columns such as upvotes and comments |
| Merge textual fields | Combines `title` and `body` into one text input |
| Clean text | Removes URLs, emojis, numbers, punctuation, and extra whitespace |
| Lowercase text | Converts text to lowercase |
| Save cleaned dataset | Stores cleaned text samples |
| Balance classes | Applies undersampling to match both labels |
| Save balanced dataset | Stores the balanced dataset for model training |

---

## Label Leakage Prevention

The `subreddit` column was removed because it can expose the label indirectly.

For example, posts from a depression-related subreddit may strongly correlate with the depressed label. If this column is kept, the model may learn subreddit identity instead of learning linguistic indicators from the text.

Removing `subreddit` makes the classification task more meaningful and forces the model to rely on post content.

---

## Text Construction

The project combines:

```text
title + body
```

into a single text field.

This is important because Reddit posts often split useful context between the title and the body. Combining both gives the model a fuller representation of the user's message.

---

## Text Cleaning

The text cleaning process includes:

```text
URL removal
Emoji and non-standard character removal
Digit removal
Punctuation removal
Line break and tab normalization
Lowercasing
Extra whitespace normalization
```

After preprocessing, the model receives cleaned textual input instead of raw Reddit content.

---

## Class Balancing

The original dataset is imbalanced, with non-depressed posts being the majority class.

To reduce model bias toward the majority class, undersampling is applied. The majority class is reduced to match the minority class.

After undersampling, the dataset becomes balanced:

```text
50% non-depressed
50% depressed
```

This helps the models learn from both classes equally.

---

## Model Development Workflow

All transformer models follow the same general workflow:

```text
Load balanced dataset
        |
        v
Split into training, validation, and test sets
        |
        v
Load model-specific tokenizer
        |
        v
Tokenize text in batches
        |
        v
Create PyTorch Dataset wrapper
        |
        v
Load pretrained transformer model
        |
        v
Fine-tune with Hugging Face Trainer
        |
        v
Evaluate on unseen test data
        |
        v
Generate classification report and confusion matrix
        |
        v
Save model locally for optional GUI inference
```

---

## Train, Validation, and Test Split

The dataset is split into three subsets:

| Split | Purpose | Percentage |
|---|---|---:|
| Training set | Model learning | 70% |
| Validation set | Tuning and early stopping | 15% |
| Test set | Final evaluation | 15% |

The split is designed to provide a fair comparison across all implemented transformer models.

---

## Models Implemented

### MiniLM

Notebook:

```text
notebooks/Mini LM Model.ipynb
```

MiniLM is a compact transformer model designed for efficient inference while maintaining strong representation quality.

In this project, MiniLM provided the best practical trade-off between performance, model size, and deployment efficiency.

---

### DistilBERT

Notebook:

```text
notebooks/distilbert_base.ipynb
```

DistilBERT is a distilled version of BERT designed to reduce model size and improve inference speed while keeping strong language understanding capabilities.

In this project, DistilBERT achieved strong classification performance and performed similarly to MiniLM at the rounded metric level.

---

### ELECTRA-small

Notebook:

```text
notebooks/electra-small-discriminator.ipynb
```

ELECTRA-small is a lightweight transformer model based on replaced-token detection.

In this project, ELECTRA-small achieved slightly lower overall performance compared to MiniLM and DistilBERT, but it remained an efficient and competitive lightweight model.

---

## Training Approach

The model notebooks use the Hugging Face training workflow.

Core training components include:

| Component | Description |
|---|---|
| Tokenizer | Converts text into model-compatible token IDs |
| PyTorch Dataset | Wraps tokenized inputs and labels |
| Sequence Classification Model | Loads a pretrained transformer with a binary classification head |
| Trainer API | Handles fine-tuning, validation, checkpointing, and evaluation |
| Early Stopping | Helps reduce overfitting |
| Evaluation per epoch | Tracks validation behavior during training |

Tokenization is performed in batches to reduce memory pressure when working with a large text dataset.

---

## Evaluation Metrics

The models are evaluated using:

| Metric | Description |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Reliability of positive predictions |
| Recall | Ability to detect actual depressed samples |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Class-level breakdown of correct and incorrect predictions |

For this task, recall and F1-score are especially important because missing depression-related posts is more sensitive than ordinary classification errors.

---

## Results

### Model Comparison

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| MiniLM | 0.95 | 0.95 | 0.95 | 0.95 |
| DistilBERT | 0.95 | 0.95 | 0.95 | 0.95 |
| ELECTRA-small | 0.94 | 0.94 | 0.94 | 0.94 |

### Comparative Conclusion

The three transformer models achieved strong and close performance.

| Model | Main Strength |
|---|---|
| MiniLM | Best balance between performance, size, and speed |
| DistilBERT | Strong general-purpose transformer baseline |
| ELECTRA-small | Lightweight and efficient with competitive results |

The final project conclusion is that MiniLM is the most balanced option because it combines strong classification capability with a compact architecture suitable for limited-resource environments.

---

## Confusion Matrix Analysis

The project includes confusion matrices for all three transformer models.

The confusion matrices help analyze:

- Correct non-depressed predictions.
- Correct depressed predictions.
- False positives.
- False negatives.

This diagnostic step is important because overall accuracy alone is not enough for mental-health-related text classification.

---

## Interpretability with LIME

The project applies LIME, Local Interpretable Model-Agnostic Explanations, to interpret model predictions.

LIME highlights the most influential words contributing to each prediction. This helps inspect whether the model is relying on meaningful linguistic indicators rather than irrelevant artifacts.

Interpretability is especially important in this project because mental-health-related text classification should not be treated as a black-box output.

---

## Gradio Demonstration

The repository includes an optional Gradio interface:

```text
notebooks/Depression Gradio (GUI).ipynb
```

The GUI allows a user to enter a Reddit-style sentence and view side-by-side predictions from:

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

The GUI notebook requires locally saved trained model folders. These model weights are intentionally not included in this repository.

---

## Model Artifacts Policy

Trained model artifacts are intentionally excluded.

Excluded files and folders include:

```text
models/
saved_model_*/
*.safetensors
*.bin
*.pt
*.pth
*.pkl
*.joblib
checkpoints/
```

This keeps the repository focused on:

- Methodology
- Preprocessing
- Training notebooks
- Evaluation
- Interpretability
- Documentation

It also prevents publishing ready-to-use model weights while still allowing the project to be reproduced by running the notebooks locally.

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the main libraries needed to run the notebooks. Version adjustments may be required depending on Python version, CUDA availability, and local environment.

### 4. Add the Dataset Locally

Create a local data folder:

```text
Data/
└── reddit_depression_dataset.csv
```

The dataset is not included in the repository.

### 5. Run Preprocessing

Open and run:

```text
notebooks/Preprocessing.ipynb
```

This notebook prepares the cleaned and balanced dataset.

### 6. Train Transformer Models

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

The GUI will not work unless the trained model folders exist locally.

---

## Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
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

- Project introduction
- Dataset overview
- Preprocessing decisions
- Class balancing
- Model development workflow
- Transformer model comparison
- Training and evaluation results
- Confusion matrices
- LIME interpretability
- Gradio demonstration
- Comparative conclusion

---

## Limitations

- The dataset is not included in the repository.
- Trained model weights are not included.
- The GUI requires local saved models before it can run.
- The project is an academic NLP experiment, not a clinical diagnostic system.
- The models classify text patterns and should not be used to diagnose depression.
- Results depend on dataset quality, labeling assumptions, and preprocessing choices.
- Reddit posts may contain sarcasm, slang, ambiguity, or incomplete context.
- Undersampling improves balance but removes many majority-class samples.
- The notebooks are experiment-oriented and are not structured as a production package.

---

## Ethical Considerations

This project deals with mental-health-related language and must be interpreted carefully.

Important constraints:

- The model output must not be treated as a medical diagnosis.
- Predictions should be interpreted as computational text classification only.
- Human review is necessary in any real mental-health setting.
- Dataset privacy and licensing must be respected.
- The system should not be used for surveillance, profiling, or automated judgment of individuals.
- False positives and false negatives may have sensitive consequences.

---

## Future Improvements

Potential improvements include:

- Add classical machine learning baselines such as TF-IDF with Logistic Regression or SVM.
- Add ROC-AUC and PR-AUC evaluation.
- Add more detailed false positive and false negative analysis.
- Add experiment tracking for model versions and metrics.
- Compare undersampling with other imbalance-handling techniques.
- Add reproducible configuration files for training.
- Export a controlled inference script without publishing trained weights.
- Refactor notebooks into a cleaner Python package.
- Add model cards for each transformer model.
- Add stronger privacy and ethical documentation.
- Add cross-dataset validation.
- Add SHAP or integrated gradients for comparison with LIME.

---

## Authors

| Name |
|---|
| Adnan Wadee Abdullah |
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

The project demonstrates a complete academic workflow: preprocessing, class balancing, transformer fine-tuning, evaluation, interpretability, and local GUI demonstration.

**Best practical model:** MiniLM  
**Top rounded performance:** 0.95 accuracy and F1-score

</div>
