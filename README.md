# US Visa Approval Prediction

This project aims to build a machine learning pipeline for predicting US visa approval. The pipeline includes stages from data ingestion to model deployment. The diagrams provided outline the workflow and components involved in the project.

# Folder Structure
![Folder Structure](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/885abe91-cdef-4833-a502-145d5fd46f40)


1. ## Project Structure
<pre>
US Visa Approval Prediction/
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── model_pusher.py
├── configuration/
│   ├── s3_operations.py
│   └── other_configurations.py
├── constant/
│   ├── constant_variables.py
├── entity/
│   ├── artifact_entity.py
│   └── config_entity.py
├── exception/
│   ├── exception_handling.py
├── logger/
│   ├── logger.py
├── pipeline/
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── utils/
│   ├── main_utils.py
├── ml/
│   ├── feature_engineering.py
│   └── models.py
├── README.md
└── other_files_and_directories
</pre>
   
2. ## Data Ingestion
- **Config Setup**: Configure paths for data ingestion directory, feature store, training/testing files, and collection name.
- **Data Ingestion**: Connect to MongoDB to retrieve data and save it to a CSV file (usvisa.csv).
- **Export to Feature Store**: The ingested data is exported to a feature store.
- **Data Splitting**: Split the data into training and testing sets, saving them as train.csv and test.csv files respectively.
- **Artifacts**: Store data ingestion artifacts for tracking.
![Data Ingestion](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/0eb36858-5a75-4f2c-899d-89d01a251efb)

3. ## Data Validation
- **Config Setup**: Configure paths for validation directories, valid/invalid data files, and drift report file path.
- **Initiate Validation**: Read the ingested data and validate the number of columns, presence of numerical and categorical columns.
- **Status Check**: Ensure columns exist in both train and test datasets, and validate the data.
- **Dataset Drift**: Check for dataset drift and produce a validation report.
![Data Validation](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/36fcacec-664e-4d96-8d0f-47b52bc93ab3)

4. ## Data Transformation
- **Config Setup**: Configure paths for transformation directories, valid/invalid train/test files, and drift report file path.
- **Initiate Transformation**: Read the train and test data, drop unnecessary columns, and apply various encoders and transformers.
- **Column Transformation**: Use one-hot encoder, ordinal encoder, power transformer, and standard scaler.
- **Data Preparation**: Prepare train and test feature arrays, apply SMOTEENN for handling imbalanced data.
- **Save Artifacts**: Store transformed data and preprocessing objects as artifacts.
![Data Transformation](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/11f3f822-5e48-4b8d-b2e1-84d80eb1762b)

5. ## Model Training
- **Config Setup**: Configure paths for model training, model file paths, expected accuracy, and model config file.
- **Model Training**: Load transformed data, split into training and testing sets, and use a neural network model factory to train the model.
- **Model Evaluation**: Evaluate the trained model against the expected accuracy, save the best model if it meets criteria.
![Model Trainer](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/ffeebbdb-92cc-4ca4-b7be-39e468bcdda4)

6. ## Model Evaluation
  - **Config Setup**: Configure paths for model evaluation, threshold score, bucket name, and S3 model key path.
- **Evaluate Models**: Compare the trained model with the best model stored in S3 based on F1 score, update if the new model performs better.
- **Save Artifacts**: Store model evaluation artifacts and responses.
![Model Evaluation](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/0f047176-1cf6-485c-b84a-4b64b0c9131c)

-
7. ## Model Pusher
- **Config Setup**: Configure paths for model pusher, S3 model key path, and bucket name.
- **Push Model**: Save the best model to the specified S3 bucket for deployment.
- **Save Artifacts**: Store model pusher artifacts for tracking.

![Model Pusher](https://github.com/nishantrv/US-Visa-Approval-Prediction/assets/18679312/096460df-aa16-4d1c-883b-71caa4a3be89)


