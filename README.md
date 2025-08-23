# AI Data Challenge: Medical Literature Classification

## Objective

The goal of this repository is to develop an Artificial Intelligence (AI) solution to classify medical literature into one or more predefined medical domains. The system will analyze the title and abstract of medical articles to predict their association with the following categories:

- **Cardiovascular**
- **Neurological**
- **Hepatorenal**
- **Oncological**

## Overview

This project aims to leverage traditional machine learning techniques to achieve accurate and explainable classification of medical articles. Specifically, we will use **Logistic Regression** as the primary model for multi-label classification. The solution will be evaluated based on its ability to correctly assign articles to the appropriate categories using the provided dataset.

## Dataset

The dataset consists of the following fields:

- **title**: The title of the medical article, which provides a concise summary of the study.
- **abstract**: A detailed summary of the article, rich in medical terminology and context.
- **group**: The target variable indicating the medical domain(s) to which the article belongs.

## Approach

The repository will implement the following steps to achieve the classification goal:

1. **Data Preprocessing**:
   - Combine the `title` and `abstract` fields into a single text field.
   - Clean the text by removing stopwords, punctuation, and applying tokenization.
   - Encode the target variable (`group`) into a binary matrix for multi-label classification.

2. **Feature Extraction**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text into numerical features.

3. **Model Training**:
   - Train a Logistic Regression model using the `OneVsRestClassifier` strategy for multi-label classification.

4. **Evaluation**:
   - Evaluate the model using metrics such as precision, recall, F1-score, and accuracy.

5. **Model Saving**:
   - Save the trained model, TF-IDF vectorizer, and label encoder for future use.

## Deliverables

- A well-documented codebase implementing the Logistic Regression solution.
- Evaluation metrics to measure the effectiveness of the model.
- Insights and justifications for the chosen approach.

## Future Work

This repository will serve as a baseline for further exploration and improvement in the field of medical literature classification. Future enhancements may include:

- Experimenting with hyperparameter tuning to optimize the Logistic Regression model.
- Exploring advanced techniques such as pre-trained language models (e.g., BioBERT) for improved performance.
- Incorporating domain-specific features using medical ontologies.

## Authors

- Juan Pablo Mejía
- Samuel Castaño
- Mateo Builes