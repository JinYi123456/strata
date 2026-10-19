# STRATA — portable container image.
#
# One process serves the API and the prebuilt console on a single port, so any
# container host works: Render, Railway, Fly.io, Cloud Run, ECS, a plain VM.
#
#   docker build -t strata .
#   docker run -p 8000:8000 strata
#
# Builds and runs on both x86-64 and arm64; there is no architecture-specific
# code, only architecture-specific speed.
#
# For the AWS Lambda deployment (arm64, S3 + DynamoDB) use deploy/Dockerfile
# instead — it has a different entry point and a smaller base image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# libgomp1 is not optional. OpenCV's parallel_for needs the OpenMP runtime, and
# without it the threshold and contour stages silently fall back to a single
# thread — correct output, a fraction of the throughput, and no error to notice.
# libgl1 and libglib2.0-0 are linked by the opencv-contrib-python wheel even when
# highgui is never called.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change does not invalidate the install layer.
COPY requirements.txt ./
COPY vision/requirements.txt vision/
COPY service/requirements.txt service/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# The neural feature weights are optional at runtime, but fetching them at build
# time means the deployed image runs the accurate backend rather than the
# fallback. A failed download is not a build failure: the pipeline degrades to
# SIFT and reports which backend it used.
RUN python scripts/fetch_models.py \
    || echo "model weights unavailable at build time; the classical backend will be used"

EXPOSE 8000

# serve.py reads PORT and binds 0.0.0.0 when it is set. A service bound to
# loopback is unreachable from outside its own container, which is a silent and
# total failure on every hosting platform.
CMD ["python", "scripts/serve.py"]
