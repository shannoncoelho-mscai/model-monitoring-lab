# Model Monitoring for Drift and Latency

## Overview

This project implements model monitoring for a machine learning model deployed as a FastAPI service on Google Cloud Run.

The experiment evaluates:

- Data drift using the Kolmogorov-Smirnov (KS) statistical test
- Model performance on drifted data
- API latency using concurrent load testing
- Cloud Run request latency using Google Cloud Monitoring
- Different types of machine learning drift
- Cloud deployment and containerization

The project was developed as part of **Lab 5: Model Monitoring for drift and latency using Cloud Monitoring Tools**.

---

## Project Objectives

The main objectives of this project are:

1. Train and deploy a machine learning model as an API.
2. Containerize the model API using Docker.
3. Deploy the containerized API to Google Cloud Run.
4. Monitor API latency under concurrent load.
5. Monitor Cloud Run request latency using Google Cloud Monitoring.
6. Create reference and current data windows.
7. Introduce controlled data drift into the current data.
8. Detect drift using a statistical test.
9. Evaluate model performance on the drifted data.
10. Interpret the relationship between data drift and model performance.

---

# 1. Machine Learning Model

## Dataset

The project uses the **Breast Cancer Wisconsin Dataset** available through scikit-learn.

The dataset contains:

- 569 samples
- 30 numerical features
- 2 target classes

The two classes are:

- malignant
- benign

## Model

The machine learning model is a pipeline consisting of:

```text
StandardScaler
      ↓
Logistic Regression
