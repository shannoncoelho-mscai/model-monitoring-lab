# Model Monitoring for Drift and Latency

## Overview

This project implements **machine learning model monitoring for data drift and API latency** using a FastAPI model service, Docker, Google Cloud Run, and Google Cloud Monitoring.

The project was developed as part of:

**Lab Assignment 5: Model Monitoring for Drift and Latency using Cloud Monitoring Tools**

The experiment focuses on monitoring a machine learning model after deployment by:

* Detecting data drift using the Kolmogorov-Smirnov (KS) statistical test
* Evaluating model performance on current data after controlled drift
* Measuring API latency using concurrent requests
* Monitoring Cloud Run request latency using Google Cloud Monitoring
* Understanding different types of machine learning drift
* Containerizing the model using Docker
* Deploying the model API to Google Cloud Run
* Using Google Artifact Registry for container image storage

---

# 1. Machine Learning Model

## Dataset

The project uses the **Breast Cancer Wisconsin Dataset** available through scikit-learn.

The dataset contains:

* **569 samples**
* **30 numerical features**
* **2 target classes**

The two target classes are:

* malignant
* benign

## Model

The machine learning model is implemented as a scikit-learn pipeline consisting of feature standardization followed by Logistic Regression.

```text
StandardScaler
      ↓
Logistic Regression
```

The model was trained using an **80/20 stratified train-test split** with:

```text
random_state = 42
```

The Logistic Regression model was configured with:

```text
max_iter = 5000
```

### Initial Test Accuracy

The model achieved:

**98.25% accuracy**

---

# 2. Project Structure

```text
model-monitoring-lab/
├── app/
│   ├── main.py
│   ├── train_model.py
│   ├── model.pkl
│   └── requirements.txt
│
├── data/
│   ├── current_data.csv
│   └── reference_data.csv
│
├── monitoring/
│   ├── create_data.py
│   ├── drift_detection.py
│   ├── evaluate_current_data.py
│   └── latency_test.py
│
├── .gitignore
├── Dockerfile
└── README.md
```

---

# 3. FastAPI Model API

The trained model is served through a **FastAPI application**.

The API provides the following endpoints:

| Endpoint   | Method | Purpose                     |
| ---------- | ------ | --------------------------- |
| `/`        | GET    | API information page        |
| `/health`  | GET    | Health check                |
| `/predict` | POST   | Generate a model prediction |
| `/docs`    | GET    | Swagger API documentation   |

The prediction endpoint accepts the 30 model features and returns:

* Prediction
* Prediction label
* Confidence
* Model inference latency

Example response:

```json
{
  "prediction": 0,
  "prediction_label": "malignant",
  "confidence": 0.98,
  "model_latency_ms": 2.31
}
```

---

# 4. Local API Testing

The FastAPI application was tested locally before containerization and deployment.

The following components were verified:

* FastAPI application startup
* Health endpoint
* Swagger documentation
* Prediction endpoint
* Model prediction response
* Model latency measurement

### Screenshots

The following screenshots were captured for the project report:

* `fastapi_home.png` — FastAPI application running locally
* `fastapi_swagger.png` — FastAPI Swagger documentation
* `api_prediction_test.png` — Successful model prediction

---

# 5. Reference and Current Data

Model monitoring requires comparison between two data windows:

### Reference Data

The reference dataset represents the distribution of data used as the baseline.

### Current Data

The current dataset represents newly observed data that is monitored for distribution changes.

The monitoring experiment generated:

* `reference_data.csv`
* `current_data.csv`

Each dataset contains **300 samples**.

The datasets contain the same model features so that their distributions can be compared.

---

# 6. Controlled Data Drift

Controlled drift was introduced into five features in the current dataset.

The affected features were:

1. Mean radius
2. Mean texture
3. Mean perimeter
4. Mean area
5. Mean smoothness

The selected features were multiplied by approximately **1.20** in the current data to create a measurable distribution shift.

This provides a controlled environment for evaluating the effectiveness of the drift detection method.

---

# 7. Drift Detection Method

The project uses the **Kolmogorov-Smirnov (KS) two-sample test** to compare the reference and current distributions.

The significance level was set to:

```text
alpha = 0.05
```

The interpretation is:

```text
p-value < 0.05
→ statistically significant distribution difference
→ drift detected
```

```text
p-value >= 0.05
→ insufficient evidence of distribution difference
→ drift not detected
```

---

# 8. Drift Detection Results

The KS test detected statistically significant drift in **5 out of 30 features**.

| Feature         | KS Statistic |    P-value | Drift |
| --------------- | -----------: | ---------: | ----- |
| mean radius     |       0.3700 | < 0.000001 | Yes   |
| mean texture    |       0.3267 | < 0.000001 | Yes   |
| mean perimeter  |       0.3567 | < 0.000001 | Yes   |
| mean area       |       0.1933 |   0.000026 | Yes   |
| mean smoothness |       0.4667 | < 0.000001 | Yes   |

The remaining features did not show statistically significant drift at the 0.05 significance level.

Therefore:

**5 / 30 features showed significant drift.**

This corresponds to approximately:

**16.67% of the monitored features.**

---

# 9. Model Performance on Current Data

The trained model was evaluated on the current dataset after introducing controlled feature drift.

### Initial Performance

```text
Accuracy = 98.25%
```

### Current Data Performance

```text
Accuracy = 95.67%
```

### Performance Difference

```text
98.25% - 95.67% = 2.58 percentage points
```

Therefore, the experiment observed a **2.58 percentage-point decrease in accuracy** after the controlled distribution shift.

The drift experiment demonstrates that changes in input feature distributions can coincide with changes in model performance.

However, drift alone does not prove that model performance degradation was caused entirely by the detected distribution shift. A labelled evaluation window is required to establish the performance impact more rigorously.

---

# 10. Docker Containerization

The FastAPI model service was containerized using Docker.

The project uses:

```text
Python 3.11
```

The Docker image installs the required Python dependencies and starts the FastAPI application using Uvicorn.

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

The container exposes the FastAPI service through the port provided by the `PORT` environment variable.

---

# 11. Local Docker Testing

The Dockerized application was tested locally before deployment to Google Cloud Run.

The health endpoint was successfully accessed from the running container.

This verified that:

* The Docker image builds successfully
* Python dependencies install correctly
* The model loads successfully
* FastAPI starts correctly
* The health endpoint responds successfully

### Screenshot

`docker_local_test.png`

**Caption:** Successful health check of the Dockerized FastAPI service.

---

# 12. Google Cloud Deployment

The containerized application was deployed to **Google Cloud Run**.

The Google Cloud project used for this experiment was:

```text
model-monitoring-lab
```

The deployment region was:

```text
asia-south1
```

The Cloud Run service was named:

```text
breast-cancer-api
```

---

# 13. Google Artifact Registry

Google Artifact Registry was used to store the Docker container image.

Repository:

```text
model-monitoring
```

Repository format:

```text
Docker
```

Repository location:

```text
asia-south1
```

Container image:

```text
breast-cancer-api
```

The image was successfully built, tagged, and pushed to Artifact Registry.

### Screenshots

* `artifact_registry_push.png`
* `artifact_registry_image.png`

**Caption for `artifact_registry_push.png`:**

Successful push of the Docker image to Google Artifact Registry.

---

# 14. Cloud Run Deployment

The Docker image stored in Artifact Registry was deployed as a Cloud Run service.

Service:

```text
breast-cancer-api
```

Region:

```text
asia-south1
```

The deployed revision was:

```text
breast-cancer-api-00001-bmb
```

The service successfully received **100% traffic**.

The deployed application was accessible through the Cloud Run URL.

### Screenshot

`cloud_run_deployment.png`

**Caption:** Successful deployment of the Dockerized FastAPI model API to Google Cloud Run.

---

# 15. Cloud Run API Validation

After deployment, the Cloud Run service was tested using the following endpoints.

### API Home

The root endpoint was accessed successfully.

Screenshot:

`cloud_run_api_home.png`

**Caption:** Breast Cancer Prediction API successfully running on Google Cloud Run.

### Health Check

The health endpoint returned a successful response.

Screenshot:

`cloud_run_health_check.png`

**Caption:** Successful health check of the FastAPI service deployed on Google Cloud Run.

### Prediction

The prediction endpoint successfully generated a model prediction.

Screenshot:

`cloud_run_prediction_test.png`

**Caption:** Successful model prediction from the FastAPI service deployed on Google Cloud Run.

---

# 16. API Latency Load Testing

API latency was measured using concurrent requests against the model prediction endpoint.

The latency experiment used:

```text
Total requests = 100
Concurrent requests = 10
```

The test measured:

* Minimum latency
* Average latency
* Median latency (P50)
* P95 latency
* Maximum latency
* Successful requests

## Client-Side Latency Results

| Metric              |      Result |
| ------------------- | ----------: |
| Total requests      |         100 |
| Successful requests |         100 |
| Minimum             |   699.84 ms |
| Average             |  1803.55 ms |
| P50                 |   883.65 ms |
| P95                 | 10008.09 ms |
| Maximum             | 10027.24 ms |

All **100 requests completed successfully**.

### Screenshot

`latency_test_results.png`

**Caption:** Concurrent API latency test results for the deployed model service.

---

# 17. Google Cloud Monitoring

Google Cloud Monitoring was used to monitor request latency for the Cloud Run service.

The monitored metric was:

```text
Cloud Run Revision
→ Request_latencies
→ Request latency
```

The monitoring period used for the experiment was:

```text
Last 1 hour
```

## Cloud Monitoring Latency Results

| Percentile |    Latency |
| ---------- | ---------: |
| P50        |   5.682 ms |
| P95        | 158.631 ms |
| P99        | 171.321 ms |

The Cloud Monitoring results provide a service-level view of request latency.

The values differ from the client-side latency test because the two measurements represent different observation points. The client-side test includes network and request overhead experienced by the load-testing client, while Cloud Monitoring provides Cloud Run request latency measurements.

### Screenshot

`cloud_monitoring_latency_percentiles.png`

**Caption:** Google Cloud Monitoring showing Cloud Run request latency percentiles.

---

# 18. Data Drift

**Data drift** occurs when the statistical distribution of input features changes between a reference dataset and a current dataset.

In this experiment, data drift was deliberately introduced into five features.

The KS test successfully identified these changes.

This demonstrates how statistical monitoring can detect changes in the input data distribution.

---

# 19. Types of Machine Learning Drift

Several types of drift are important in production machine learning systems.

## 19.1 Data Drift / Covariate Shift

Data drift occurs when the distribution of input variables changes.

```text
P(X) changes
```

Example:

A medical model receives patient measurements with a significantly different distribution from the training or reference population.

---

## 19.2 Label Shift

Label shift occurs when the distribution of the target variable changes.

```text
P(Y) changes
```

Example:

The proportion of malignant and benign cases changes significantly in the monitored population.

---

## 19.3 Concept Drift

Concept drift occurs when the relationship between the input variables and the target changes.

```text
P(Y | X) changes
```

Example:

The relationship between patient measurements and disease outcome changes over time.

---

## 19.4 Prediction Drift

Prediction drift occurs when the distribution of model predictions changes.

For example, a model may begin producing a much higher proportion of one prediction class than it did during the reference period.

---

## 19.5 Domain Shift

Domain shift occurs when the environment or population represented by the current data differs from the environment represented by the training data.

Examples include:

* Different hospitals
* Different geographical populations
* Different equipment
* Different data collection procedures

---

# 20. Latency and Cold Starts

Cloud Run applications can experience additional latency when a new container instance needs to start.

This is commonly known as a **cold start**.

Latency can be influenced by:

* Container startup time
* Model loading
* Network overhead
* Request concurrency
* CPU allocation
* Instance scaling
* Application initialization

The latency experiment showed that most requests were substantially faster than the maximum observed latency, while a small number of requests experienced very high latency.

The P95 and maximum values therefore provide useful information about tail latency that would not be visible from average latency alone.

---

# 21. Overall Experimental Results

The major results of the experiment are summarized below.

| Component             | Result                               |
| --------------------- | ------------------------------------ |
| Dataset               | Breast Cancer Wisconsin              |
| Samples               | 569                                  |
| Features              | 30                                   |
| Model                 | StandardScaler + Logistic Regression |
| Initial accuracy      | 98.25%                               |
| Current-data accuracy | 95.67%                               |
| Accuracy difference   | 2.58 percentage points               |
| Drifted features      | 5 / 30                               |
| Drift method          | KS two-sample test                   |
| Significance level    | 0.05                                 |
| Load-test requests    | 100                                  |
| Concurrent requests   | 10                                   |
| Client P50            | 883.65 ms                            |
| Client P95            | 10008.09 ms                          |
| Cloud Monitoring P50  | 5.682 ms                             |
| Cloud Monitoring P95  | 158.631 ms                           |
| Cloud Monitoring P99  | 171.321 ms                           |
| Deployment            | Google Cloud Run                     |

---

# 22. Recommended Monitoring Action

The results suggest that the deployed model should continue to be monitored rather than immediately retrained solely because drift was detected.

A suitable production monitoring strategy would include:

1. Monitor feature distributions continuously.
2. Monitor model prediction distributions.
3. Monitor model performance when labelled data becomes available.
4. Monitor P50, P95, and P99 latency.
5. Investigate sustained or increasing drift.
6. Investigate significant performance degradation.
7. Retrain the model when the detected changes have meaningful operational impact.

A practical retraining decision should consider both **data drift** and **model performance**, rather than relying on drift detection alone.

---

# 23. Technologies Used

## Machine Learning

* Python
* scikit-learn
* Logistic Regression
* StandardScaler
* Breast Cancer Wisconsin Dataset

## API

* FastAPI
* Uvicorn
* Pydantic

## Monitoring

* SciPy
* Kolmogorov-Smirnov statistical test
* Concurrent latency testing
* Google Cloud Monitoring

## Containerization

* Docker
* Docker Desktop

## Cloud Deployment

* Google Cloud Run
* Google Artifact Registry
* Google Cloud Monitoring
* Google Cloud Logging
* Google Cloud Build

## Development Tools

* Visual Studio Code / Cursor
* PowerShell
* Git
* GitHub

---

# 24. Google Cloud Configuration

The following Google Cloud services were enabled for the project:

* Cloud Run Admin API
* Artifact Registry API
* Cloud Build API
* Cloud Logging API
* Cloud Monitoring API

### Project

```text
Project ID: model-monitoring-lab
```

### Region

```text
asia-south1
```

### Artifact Registry

```text
Repository: model-monitoring
Format: Docker
Region: asia-south1
```

### Cloud Run

```text
Service: breast-cancer-api
Region: asia-south1
```

### Screenshot

`gcp_project_id.png`

**Caption:** Google Cloud project configuration showing the project used for the model monitoring experiment.

---

# 25. Billing and Cost Control

Google Cloud services were used for the deployment and monitoring experiment.

A billing budget was configured with a limit of:

```text
$10
```

Budget alert thresholds were configured at:

* 50%
* 75%
* 90%
* 100%

These alerts help prevent unexpected cloud expenditure during experimentation.

The deployed resources should be stopped, deleted, or scaled appropriately after the lab when they are no longer required.

---

# 26. Security Considerations

The project follows basic security practices for the lab deployment.

The `.gitignore` file excludes:

```text
.env
.env.*
*.json
```

This helps prevent local environment files and Google Cloud credential files from being committed accidentally.

The trained model file is explicitly allowed:

```text
!app/model.pkl
```

The project does not store API keys, passwords, or cloud credentials in the repository.

For a production deployment, additional controls would be recommended, including:

* Authentication and authorization
* HTTPS enforcement
* Secret Manager
* IAM-based access control
* Rate limiting
* Request validation
* Audit logging
* Monitoring alerts

---

# 27. Screenshots

The following screenshots were captured during the experiment and are included in the project report.

| Screenshot                                 | Description                                |
| ------------------------------------------ | ------------------------------------------ |
| `gcp_project_id.png`                       | Google Cloud project configuration         |
| `fastapi_home.png`                         | FastAPI application running locally        |
| `fastapi_swagger.png`                      | FastAPI Swagger documentation              |
| `api_prediction_test.png`                  | Successful local model prediction          |
| `drift_data_creation.png`                  | Creation of reference and current datasets |
| `drift_detection_summary.png`              | KS-test drift detection results            |
| `current_data_evaluation.png`              | Model evaluation on current data           |
| `docker_local_test.png`                    | Dockerized API health check                |
| `artifact_registry_push.png`               | Docker image pushed to Artifact Registry   |
| `artifact_registry_image.png`              | Artifact Registry image                    |
| `cloud_run_deployment.png`                 | Cloud Run deployment                       |
| `cloud_run_api_home.png`                   | Cloud Run API home page                    |
| `cloud_run_health_check.png`               | Cloud Run health check                     |
| `cloud_run_prediction_test.png`            | Cloud Run prediction test                  |
| `latency_test_results.png`                 | Client-side latency test                   |
| `cloud_monitoring_latency_percentiles.png` | Cloud Monitoring latency percentiles       |

---

# 28. Monitoring and Deployment Summary

The complete workflow implemented in this project is:

```text
Breast Cancer Dataset
        ↓
Train ML Model
        ↓
Evaluate Initial Accuracy
        ↓
Save model.pkl
        ↓
FastAPI Prediction API
        ↓
Local API Testing
        ↓
Docker Containerization
        ↓
Docker Local Testing
        ↓
Google Artifact Registry
        ↓
Google Cloud Run
        ↓
Cloud Run API Validation
        ↓
Concurrent Latency Testing
        ↓
Google Cloud Monitoring
        ↓
Reference vs Current Data
        ↓
Controlled Data Drift
        ↓
KS Statistical Test
        ↓
Model Evaluation on Current Data
        ↓
Interpret Monitoring Results
```

---

# 29. Conclusion

This project demonstrates an end-to-end machine learning monitoring workflow covering **model deployment, data drift detection, model performance evaluation, API latency measurement, containerization, and cloud monitoring**.

A Logistic Regression model trained on the Breast Cancer Wisconsin dataset achieved an initial accuracy of **98.25%**. Controlled changes were introduced into five input features, and the Kolmogorov-Smirnov test successfully detected statistically significant drift in those five features.

When evaluated on the current data, model accuracy decreased to **95.67%**, representing a **2.58 percentage-point decrease** from the initial accuracy.

The model API was successfully containerized using Docker and deployed to **Google Cloud Run** through **Google Artifact Registry**. The API successfully handled the concurrent test workload, with 100 out of 100 requests completing successfully.

Client-side latency testing produced a **P50 of 883.65 ms** and a **P95 of 10008.09 ms**, while Google Cloud Monitoring reported Cloud Run request latency percentiles of **5.682 ms at P50, 158.631 ms at P95, and 171.321 ms at P99**.

Overall, the experiment demonstrates why production machine learning systems should monitor both **data distribution and service performance**. Detecting drift early, together with tracking model accuracy and latency, can help identify situations where further investigation, model retraining, or deployment changes may be required.
