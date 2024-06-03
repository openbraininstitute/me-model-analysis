# me-model-analysis

### Repo that contains the code to run me-model-analysis using on-demand service in AWS

- It will be deployed in the docker hub image `bluebrain/me-model-analysis`
- It will be consumed by the frontend when we launch a validation of the single cell me-model.
- It will use the on-demand service in AWS https://bbpgitlab.epfl.ch/cs/cloud/aws/deployment/-/merge_requests/437

# Build

### Download necessary packages
```
mkdir -p packages && pushd packages
git clone --depth 1 https://bbpgitlab.epfl.ch/msg/icselector.git icselector
git clone https://github.com/BlueBrain/nexus-forge.git nexusforge
git clone --depth 1 https://github.com/BlueBrain/BluePyEModel.git bluepyemodel
git clone --depth 1 https://bbpgitlab.epfl.ch/cells/bluepyemodelnexus bluepyemodelnexus
popd
```

### Build package
```
make docker_build
```

### Push package
You need the proper dockerhub credentials
```
ORIGINAL_IMG="me-model-analysis:dev"
END_IMG="bluebrain/me-model-analysis:latest"

docker tag $ORIGINAL_IMG $END_IMG
docker push $END_IMG
```
