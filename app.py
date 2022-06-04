from dotenv import load_dotenv
import os
import datarobot as dr
import pandas as pd

load_dotenv('.env')

dr.Client()
# Error kicked by calling for Project.list() with no projects on account
# dr.Project.list()

# Example code below pulled from DR Tutorials

# Set to the location of your project.csv and project-test.csv data files
# Example: dataset_file_path = '/Users/myuser/Downloads/project-test.csv'
training_dataset_file_path = ''
test_dataset_file_path = ''

# Load dataset
training_dataset = dr.Dataset.create_from_file(training_dataset_file_path)
# ALTERNATIVELY - Load dataset from Pandas DataFrame
training_data_df = pd.read_csv(training_dataset_file_path)
training_dataset = dr.Dataset.create_from_in_memory_data(training_data_df)

# Create a new project based on dataset
project = dr.Project.create_from_dataset(training_dataset.id, project_name='Project Name DR-Client')

# Set target for dataset training
project.set_target('some target')

# Explore dataset
all_features = training_dataset.get_all_features()
for feature in all_features:
    print(feature.name, feature.feature_type, feature.dataset_id)
    print(feature.max, feature.mean, feature.min, feature.std_dev)
    print(feature.na_count, feature.unique_count)
    # optional:
    # feature.get_histogram().plot

# Use training data to build models.
project.wait_for_autopilot()
model = dr.ModelRecommendation.get(project.id).get_model()

# By default, models are evaluated on the first validation partition. 
# To start cross-validation, use the Model.cross_validate method:
model_job_id = model.cross_validate()

# Check which features were actually used by the model - not all are guaranteed
feature_names = model.get_features_used()

# Test predictions on new data
prediction_data = project.upload_dataset(test_dataset_file_path)
predict_job = model.request_predictions(prediction_data.id)
predictions = predict_job.get_result_when_complete()
predictions.head()

# Deployment steps for Deployment via AI Platform Trial
deployment = dr.Deployment.create_from_learning_model(
    model_id=model.id, label="MPG Prediction Server",
    description="Deployed with DataRobot client")

# View deployment stats
service_stats = deployment.get_service_stats()
print(service_stats.metrics)


# Make predictions against the deployed model
import requests
from pprint import pprint
import json
import os
# JSON records for example autos for which to predict mpg
autos = [
    {
        "cylinders": 4,
        "displacement": 119.0,
        "horsepower": 82.00,
        "weight": 2720.0,
        "acceleration": 19.4,
        "model year": 82,
        "origin": 1,
    },
    {
        "cylinders": 8,
        "displacement": 120.0,
        "horsepower": 79.00,
        "weight": 2625.0,
        "acceleration": 18.6,
        "model year": 82,
        "origin": 1,
    },
]
# Create REST request for prediction API
prediction_server = deployment.default_prediction_server
prediction_headers = {
    "Authorization": "Bearer {}".format(os.getenv("DATAROBOT_API_TOKEN")),
    "Content-Type": "application/json",
    "datarobot-key": prediction_server['datarobot-key']
}

predictions = requests.post(
    f"{prediction_server.url}/predApi/v1.0/deployments"
    f"/{deployment.id}/predictions",
    headers=prediction_headers,
    data=json.dumps(autos),
)
pprint(predictions.json())
