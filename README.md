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

The model was trained using an 80/20 stratified train-test split with:

random_state = 42

The Logistic Regression model was configured with:

max_iter = 5000
Initial Test Accuracy
98.25%
2. Project Structure
model-monitoring-lab/
│
├── app/
│   ├── main.py
│   ├── train_model.py
│   ├── model.pkl
│   └── requirements.txt
│
├── monitoring/
│   ├── create_data.py
│   ├── drift_detection.py
│   ├── evaluate_current_data.py
│   └── latency_test.py
│
├── data/
│   ├── reference_data.csv
│   └── current_data.csv
│
├── tests/
│
├── Dockerfile
├── README.md
└── .gitignore
3. FastAPI Application

The trained machine learning model was exposed through a FastAPI application.

Available endpoints:

GET  /
GET  /health
POST /predict
GET  /docs

The /predict endpoint accepts 30 numerical features and returns:

predicted class
prediction label
confidence
model latency

The FastAPI application was first tested locally before containerization.

4. Local API Testing

The FastAPI application was run locally using Uvicorn.

The application was tested using:

FastAPI home page
Swagger API documentation
Health endpoint
Prediction endpoint

The Swagger documentation was available at:

/docs

A valid 30-feature sample was submitted to the prediction endpoint.

The API successfully returned a model prediction with HTTP status code 200.

5. Data Drift Experiment

Two data windows were created for the drift experiment.

Reference Data
300 samples
30 features
Current Data
300 samples
30 features

Controlled drift was introduced into five features of the current dataset.

The values of these features were multiplied by:

1.20

This represents a controlled 20% increase.

The affected features were:

mean radius
mean texture
mean perimeter
mean area
mean smoothness

The remaining 25 features were not intentionally modified.

6. Statistical Drift Detection

The two-sample Kolmogorov-Smirnov (KS) test was used to compare the distributions of the reference and current datasets.

The significance level was:

alpha = 0.05

A feature was considered to have statistically significant drift when:

p-value < 0.05
Drift Detection Result

The KS test evaluated:

30 features

Statistically significant drift was detected in:

5 out of 30 features

The detected features were:

mean radius
mean texture
mean perimeter
mean area
mean smoothness

The five intentionally modified features were successfully detected as drifted.

The remaining 25 features did not show statistically significant drift.

KS Test Results
Feature	KS Statistic	p-value
mean radius	0.3700	<0.000001
mean texture	0.3267	<0.000001
mean perimeter	0.3567	<0.000001
mean area	0.1933	0.000026
mean smoothness	0.4667	<0.000001
7. Model Performance on Current Data

Statistical drift alone does not prove that model performance has degraded.

Therefore, the model was evaluated separately on the current data using its true labels.

Original Test Accuracy
98.25%
Accuracy on Current Drifted Data
95.67%
Accuracy Difference
2.58 percentage points

Therefore:

Original accuracy       = 98.25%
Current accuracy        = 95.67%
Accuracy decrease       = 2.58 percentage points

In this controlled experiment, the detected input distribution changes were accompanied by a measurable decrease in model accuracy.

8. Interpretation of Drift Results

The detected drift represents data drift because the distribution of the input features changed between the reference and current data windows.

The drift was intentionally introduced by increasing five selected features by 20%.

However, data drift does not automatically mean that a machine learning model has become worse.

To determine whether the model was affected, the current data was evaluated using its true labels.

In this experiment, model accuracy decreased from 98.25% to 95.67%.

Therefore, the controlled feature distribution change was accompanied by a measurable decrease in predictive performance.

In a real production system, persistent drift should be investigated together with model performance before deciding whether retraining is required.

9. Types of Machine Learning Drift
Data Drift

Data drift occurs when the distribution of input features changes between a reference period and a current period.

This is the type of drift directly measured in this experiment.

Label Shift

Label shift occurs when the distribution of target classes changes while the relationship between the input features and target remains approximately unchanged.

For example, the proportion of malignant and benign cases could change over time without changing the underlying relationship between the features and the target.

Concept Drift

Concept drift occurs when the relationship between the input features and the target variable changes over time.

In this situation, the same input patterns may correspond to different outcomes compared with the period in which the model was trained.

Prediction Drift

Prediction drift occurs when the distribution of model predictions changes between a reference period and a current period.

For example, a classification model may begin producing a substantially different proportion of predictions for one class compared with its historical prediction distribution.

Domain Shift

Domain shift occurs when a model is applied in an environment or domain that differs from the environment represented by its training data.

For example, a model trained using one type of data may behave differently when applied to data from another environment with different characteristics.

10. Docker Containerization

The FastAPI application was containerized using Docker.

The Docker image used Python 3.11 and the required machine learning dependencies.

Scikit-learn was pinned to version 1.9.1 to maintain compatibility with the trained model environment.

The Docker image was built as:

breast-cancer-api:latest

The container was tested locally before deployment.

The local Docker container successfully started the FastAPI application on port 8080.

The health endpoint returned:

status
------
healthy

This confirmed that the Dockerized application was running correctly.

11. Google Artifact Registry

The Docker image was uploaded to Google Artifact Registry.

GCP Project
model-monitoring-lab
Artifact Registry Repository
model-monitoring
Region
asia-south1
Docker Image
breast-cancer-api:latest

The image was stored using the following Artifact Registry path:

asia-south1-docker.pkg.dev/model-monitoring-lab/model-monitoring/breast-cancer-api:latest
12. Google Cloud Run Deployment

The Dockerized FastAPI application was deployed to Google Cloud Run.

Cloud Run Service
breast-cancer-api
Region
asia-south1
GCP Project
model-monitoring-lab

The deployed Cloud Run service successfully received traffic.

The deployed service URL was:

https://breast-cancer-api-1037614303083.asia-south1.run.app

The Cloud Run deployment successfully created revision:

breast-cancer-api-00001-bmb

100% of the traffic was routed to the deployed revision.

13. Cloud Run API Validation

The deployed API was tested using its public Cloud Run URL.

The following endpoints were validated:

/
 /health
/predict

The health endpoint returned:

status
------
healthy

The prediction endpoint was also tested using a valid 30-feature input.

Example result:

prediction       : 0
prediction_label : malignant
confidence       : 0.9999999938818932
model_latency_ms : 11.809

This confirmed that the complete deployed machine learning pipeline was operational.

14. Latency Load Testing

A client-side load test was performed against the deployed Cloud Run prediction API.

The load test generated:

Total requests: 100
Concurrent requests: 10
Successful requests: 100

Each request contained a valid 30-feature input.

Latency was measured from immediately before sending the HTTP request until the response was received.

Therefore, these measurements represent end-to-end client-side latency, including network communication and Cloud Run request processing.

Client-Side Latency Results
Minimum latency: 699.84 ms
Average latency: 1803.55 ms
P50 latency: 883.65 ms
P95 latency: 10008.09 ms
Maximum latency: 10027.24 ms

The P50 latency was 883.65 ms.

The P95 latency was 10008.09 ms.

The large difference between P50 and P95 indicates a substantial high-latency tail under the tested concurrent workload.

15. Cloud Monitoring Latency

Google Cloud Monitoring was used to observe the deployed Cloud Run service.

Cloud Run automatically integrates with Google Cloud Monitoring.

The Cloud Run request latency metric was selected using Metrics Explorer.

Percentile aggregation was used to observe the latency distribution.

Cloud Monitoring Results
P50 latency: 5.682 ms
P95 latency: 158.631 ms
P99 latency: 171.321 ms

The P50 value indicates that approximately half of the observed requests had latency below 5.682 ms.

The P95 value indicates that approximately 95% of the observed requests had latency below 158.631 ms.

The P99 value was 171.321 ms.

16. Client-Side Latency vs Cloud Monitoring Latency

The client-side latency measurements and Cloud Monitoring measurements represent different parts of the request path.

The Python load-testing script measured end-to-end client-side latency.

This includes:

Network communication
Request transmission
Cloud Run processing
Response transmission
Other external effects

Cloud Monitoring provides a service-side view of Cloud Run request latency.

Therefore, the two measurements are not expected to be identical.

The client-side test showed:

P50 = 883.65 ms
P95 = 10008.09 ms

Cloud Monitoring showed:

P50 = 5.682 ms
P95 = 158.631 ms
P99 = 171.321 ms

The difference demonstrates why both application-level and cloud-level monitoring can be useful when investigating production latency.

17. Cold Starts and High Tail Latency

Cloud Run can start a new container instance when requests arrive after a period of inactivity.

This behaviour can result in additional startup latency and is commonly referred to as a cold start.

The latency experiment did not remove slow requests or exclude possible cold-start requests.

Therefore, the reported P95 includes the complete set of measured requests.

The client-side results showed a large difference between P50 and P95.

Possible contributors to the high-latency tail include:

Cloud Run container startup behaviour
Network communication
Request scheduling
Concurrent request handling

For a production system, repeated high tail latency should be investigated further.

Possible actions include:

Reviewing Cloud Run configuration
Investigating cold starts
Adjusting concurrency
Reducing request processing time
Considering minimum instances when consistently low startup latency is required
18. Overall Results
Drift Results
Reference samples: 300
Current samples: 300
Features tested: 30
Features with significant drift: 5
Significance level: 0.05
Model Performance
Original accuracy: 98.25%
Current drifted-data accuracy: 95.67%
Accuracy decrease: 2.58 percentage points
Client-Side Latency
Total requests: 100
Concurrency: 10
Successful requests: 100
P50: 883.65 ms
P95: 10008.09 ms
Cloud Monitoring Latency
P50: 5.682 ms
P95: 158.631 ms
P99: 171.321 ms
19. Recommended Action

The experiment successfully demonstrated both data drift detection and operational latency monitoring for a machine learning model deployed on Google Cloud Run.

The KS test successfully detected the five intentionally modified features.

The model evaluation showed that the controlled distribution changes were accompanied by a decrease in accuracy from 98.25% to 95.67%.

If this were a production system, the recommended approach would be to continue monitoring:

affected input features
model performance
prediction behaviour
API latency
high-latency requests

Persistent drift should be investigated to determine its cause.

If the distribution change continues and causes unacceptable degradation in predictive performance, the model should be retrained using representative recent data.

Persistent high latency should also be investigated by examining:

network effects
concurrency
container startup behaviour
Cloud Run configuration
20. Technologies Used
Python
FastAPI
Uvicorn
scikit-learn
Pandas
NumPy
SciPy
Joblib
Docker
Google Cloud Run
Google Artifact Registry
Google Cloud Monitoring
Google Cloud Logging
PowerShell
Git/GitHub
21. Monitoring and Deployment Summary
Component	Technology
Dataset	Breast Cancer Wisconsin
Model	Logistic Regression
Preprocessing	StandardScaler
API	FastAPI
Local Server	Uvicorn
Containerization	Docker
Container Registry	Google Artifact Registry
Deployment	Google Cloud Run
Cloud Region	asia-south1
Drift Test	Kolmogorov-Smirnov
Drift Significance Level	0.05
Load Test	Python ThreadPoolExecutor
Cloud Monitoring	Google Cloud Monitoring
GCP Project	model-monitoring-lab
22. Screenshots

The project documentation contains screenshots demonstrating the major stages of the experiment.

Important screenshots include:

fastapi_home.png
fastapi_swagger.png
api_prediction_test.png
drift_data_creation.png
drift_detection_summary.png
current_data_evaluation.png
docker_local_test.png
artifact_registry_push.png
artifact_registry_image.png
cloud_run_deployment.png
cloud_run_api_home.png
cloud_run_health_check.png
cloud_run_prediction_test.png
latency_test_results.png
cloud_monitoring_latency_percentiles.png
gcp_project_id.png

These screenshots document:

Local FastAPI execution
API documentation
Local prediction testing
Drift dataset creation
KS drift detection
Current-data model evaluation
Docker testing
Artifact Registry upload
Cloud Run deployment
Cloud Run API validation
Client-side latency testing
Cloud Monitoring latency
GCP project identification
23. GCP Configuration

GCP Project ID:

model-monitoring-lab

Cloud Run Region:

asia-south1

Cloud Run Service:

breast-cancer-api

Artifact Registry Repository:

model-monitoring

The project was configured with the required Google Cloud services for deployment and monitoring.

The main APIs used include:

Cloud Run Admin API
Artifact Registry API
Cloud Build API
Cloud Logging API
Cloud Monitoring API
24. Billing and Cost Control

A Google Cloud billing budget alert was configured for the project.

The budget amount was:

$10

Budget thresholds were configured at:

50%
75%
90%
100%

This provides an additional safeguard against unexpected cloud usage during the experiment.

25. Security Considerations

No passwords, API keys, private credentials, or service-account key files should be committed to the project repository.

The .gitignore file excludes common local and sensitive files such as:

.venv/
__pycache__/
*.pyc
.env
.env.*

Cloud credentials should remain outside the project source code.

26. Conclusion

This project implemented a complete model monitoring workflow for a machine learning API.

The workflow included:

Dataset
   ↓
Model Training
   ↓
FastAPI API
   ↓
Docker Container
   ↓
Artifact Registry
   ↓
Google Cloud Run
   ↓
Latency Monitoring
   ↓
Data Drift Detection
   ↓
Model Performance Evaluation

The controlled drift experiment successfully detected five drifted features using the Kolmogorov-Smirnov statistical test.

The model accuracy decreased from 98.25% on the original test data to 95.67% on the drifted current data.

The latency experiment demonstrated the difference between client-side end-to-end latency and Cloud Monitoring request latency.

Overall, the project demonstrates how model behaviour and operational performance can be monitored after deploying a machine learning model to the cloud.


**Important:** In your current file, everything after:

```text
StandardScaler
        ↓
Logistic Regression