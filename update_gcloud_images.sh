# Should update, or make this happen automatically using Cloud Build.
# For now, use this to update the images that gcloud uses for Cloud Run

# If not working, run commands manually

# Login first using: gcloud auth login

# authenticate repo, should be done once
gcloud auth configure-docker europe-west2-docker.pkg.dev

# tag images
docker tag jobsfrontend-frontend europe-west2-docker.pkg.dev/ttds-416315/ttds-repository/jobs-frontend
docker tag jobsfrontend-backend europe-west2-docker.pkg.dev/ttds-416315/ttds-repository/jobs-backend

# push
docker push europe-west2-docker.pkg.dev/ttds-416315/ttds-repository/jobs-frontend
docker push europe-west2-docker.pkg.dev/ttds-416315/ttds-repository/jobs-backend