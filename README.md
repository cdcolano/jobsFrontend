## Text Technologies for Data Science

# Web Search

## Backend

### GCP
To Deploy to GCP build Docker image locally and push to remote repository.

> Be sure to upgrade the tag/version number according to semver versioning standards

- Build:
```shell
    # be sure to be on the root of the repo
    cd jobsFrontend
    # Run the command to build the docker image
    docker build . --rm \
      -t "${BACKEND_IMG_NAME}:${BACKEND_IMG_TAG}" \
      -t "${BACKEND_IMG_NAME}:latest" \
      -f ./services/backend/Dockerfile \
      --platform linux/amd64
```
_**Include the --platform flag if building in a Mac with M1 processor, this is just for deploying 
--platform tag is not needed if building for local dev**_

- Tag:
```shell
   docker image tag "${BACKEND_IMG_NAME}:${BACKEND_IMG_TAG}" \
    "${GCP_DOCKER_LOCATION}/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/${BACKEND_IMG_NAME}:${BACKEND_IMG_TAG}"      
```

- Push:
```shell
  docker push -a "${GCP_DOCKER_LOCATION}/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/${BACKEND_IMG_NAME}"
```

Continue over on GCP after pushing built image.


## Frontend

### GCP
To Deploy to GCP build Docker image locally and push to remote repository.

> Be sure to upgrade the tag/version number according to semver versioning standards

- Build:
```shell
    # be sure to be on the frontend service folder
    cd ./services/frontend
    # Run the command to build the docker image
    docker build . --rm \
      -t "${FRONTEND_IMG_NAME}:${FRONTEND_IMG_TAG}" \
      -t "${FRONTEND_IMG_NAME}:latest" \
      --platform linux/amd64
```
_**Include the --platform flag if building in a Mac with M1 processor, this is just for deploying 
--platform tag is not needed if building for local dev**_

- Tag:
```shell
   docker image tag "${FRONTEND_IMG_NAME}:${FRONTEND_IMG_TAG}" \
    "${GCP_DOCKER_LOCATION}/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/${FRONTEND_IMG_NAME}:${FRONTEND_IMG_TAG}"      
```

- Push:
```shell
  docker push -a "${GCP_DOCKER_LOCATION}/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/${FRONTEND_IMG_NAME}"
```

Continue over on GCP after pushing built image.


## GCP
For deployments on GCP go to the cloudrun console and either create a new service or modify an existing one.

Be sure to select the latest tag/version of the image.

Be sure to include the required environment variables for each service.
