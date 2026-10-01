# Inference image for the treatment effect model.
#
# Build (from the project root):
#   docker build -t treatment-effect -f docker/Dockerfile .
#
# Batch scoring (mount a folder with the input json; predictions land next to it):
#   docker run --rm -v "$PWD/data:/data" treatment-effect \
#       --input /data/inference_inputs.json --output /data/predictions.csv
#
# HTTP service:
#   docker run --rm -p 8000:8000 treatment-effect serve
#   curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @patients.json

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

WORKDIR /app

COPY docker/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/treatment_effect /app/treatment_effect
COPY artifacts/model.joblib artifacts/model.json /app/artifacts/

# Fail the build early if the model does not load in this environment.
RUN python -c "from treatment_effect.models import TreatmentEffectModel; m = TreatmentEffectModel.load('artifacts/model.joblib'); print('model ok:', m.learner.name, 'horizon', m.horizon)"

EXPOSE 8000
ENTRYPOINT ["python", "-m", "treatment_effect.predict", "--model", "/app/artifacts/model.joblib"]
CMD ["--help"]
