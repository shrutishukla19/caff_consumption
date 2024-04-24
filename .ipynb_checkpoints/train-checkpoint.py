import os, joblib, logging, argparse
import pandas as pd
import numpy as np
# from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
# from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

#Input Arguments
parser.add_argument(
    '--data_gcs_path',
    help = 'Dataset file on Google Cloud Storage',
    type = str
)

parser.add_argument(
    '--model_dir',
    help = 'Directory to output model artificats',
    type = str,
    default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
)

#Parse arguments
args = parser.parse_args()
arguments = args.__dict__

#Get dataset from GCS

data_gcs_path = arguments['data_gcs_path']
df = pd.read_csv(data_gcs_path)
logging.info("reading gs data: {}".format(data_gcs_path))

#label encoding
columns = 'Caff'
le = LabelEncoder()
df[columns] = le.fit_transform(df[columns])

#split data into feature and target
X = df.drop(columns = columns)
y = df[columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest Classifier
# Create an instance of RandomForestClassifier
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Score the model
score = model.score(X_test, y_test)
preds = model.predict(X_test)

print("Score: " ,score)
# print("Predictions: " ,preds)

# Print classification report
report = classification_report(y_test, model.predict(X_test))
print(report)
logging.info("Classification Report for RandomForestClassifier - Caff")
logging.info(report)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
logging.info('Accuracy: {}'.format(acc))

# # calculate AUC
# y_scores = model.predict_proba(X_test)
# auc = roc_auc_score(y_test,y_scores[:,1])
# print('AUC: ' + str(auc))
# logging.info('AUC: {}'.format(auc))

artifact_filename = 'model.joblib'

# save model artifact to local filesystem
local_path = artifact_filename
joblib.dump(model, local_path)

# Upload model artifact to cloud storage
model_directory = arguments['model_dir']
if model_directory == "":
  print("Training is run locally - skipping model saving to GCS.")
else:
  storage_path = os.path.join(model_directory, artifact_filename)
  blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
  blob.upload_from_filename(local_path)
  logging.info("modelexported to : {}".format(storage_path))