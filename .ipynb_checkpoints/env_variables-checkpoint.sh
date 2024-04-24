export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=drug-caff-docker-repo
export IMAGE_NAME=drug_caff_model
export IMAGE_TAG=latest
export IMAGE_URI=us-central1-a-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}