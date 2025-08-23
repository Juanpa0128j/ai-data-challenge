# AI Data Challenge: Medical Literature Classification

## Objective

The goal of this repository is to develop an Artificial Intelligence (AI) solution to classify medical literature into one or more predefined medical domains. The system will analyze the title and abstract of medical articles to predict their association with the following categories:

- **Cardiovascular**
- **Neurological**
- **Hepatorenal**
- **Oncological**

## Overview

This project aims to leverage advanced AI techniques, including traditional machine learning, language models, or hybrid approaches, to achieve accurate and explainable classification of medical articles. The solution will be evaluated based on its ability to correctly assign articles to the appropriate categories using the provided dataset.

## Dataset

The dataset consists of the following fields:

- **title**: The title of the medical article, which provides a concise summary of the study.
- **abstract**: A detailed summary of the article, rich in medical terminology and context.
- **group**: The target variable indicating the medical domain(s) to which the article belongs.

## Approach

The repository will explore various strategies to achieve the classification goal, including:

1. **Traditional Machine Learning**: Using techniques like TF-IDF and logistic regression.
2. **Pre-trained Language Models**: Fine-tuning models like BioBERT or PubMedBERT.
3. **Hybrid Approaches**: Combining domain-specific features with contextual embeddings.
4. **Large Language Models (LLMs)**: Utilizing models like GPT for zero-shot or few-shot classification.

## Deliverables

- A well-documented codebase implementing the chosen solution(s).
- Evaluation metrics to measure the effectiveness of the model.
- Insights and justifications for the chosen approach.

## Future Work

This repository will serve as a baseline for further exploration and improvement in the field of medical literature classification. Contributions and suggestions are welcome to enhance the solution.

## Authors

- Juan Pablo Mejía
- Samuel Castaño
- Mateo Builes