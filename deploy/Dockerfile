# STRATA vision workload — AWS Lambda container (arm64 / Graviton)
#
# The vision core is a pure function of its inputs, so it containerises without
# a conditional anywhere: the same modules that run in the console run here.
#
# Build for Arm explicitly. On an x86 host that means buildx:
#   docker buildx build --platform linux/arm64 -t strata-vision .

FROM public.ecr.aws/lambda/python:3.12-arm64

# OpenMP runtime for OpenCV's parallel_for; the Lambda base image does not
# carry it, and without it the contour and threshold stages run single-threaded
# (which quietly halves throughput rather than failing).
RUN dnf install -y libgomp && dnf clean all

COPY vision/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall \
      opencv-contrib-python-headless==5.0.0.93
# That last line swaps the flavour, not the version. The GUI wheel that
# requirements.txt names links libxcb and the rest of the X11 client libraries,
# which neither this base image nor the zip-deployed runtime carries, so `import
# cv2` dies with "libxcb.so.1: cannot open shared object file" and every
# invocation fails before it starts. -headless is the same 5.0.0.93 build with
# the windowing code left out - and this project never opens a window. It was
# found by running the zip path (deploy/deploy.py), which installs headless for
# exactly that reason; this image has not been rebuilt here because this machine
# has no Docker, so treat this line as unverified until someone builds it.

COPY vision/ ${LAMBDA_TASK_ROOT}/
COPY deploy/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# Feature weights are not committed. A build from a working copy carries them in
# through the COPY above; a build from a fresh clone does not, and the pipeline
# falls back to SIFT. Say which happened at build time rather than leaving it to
# be discovered from a latency graph later — the handler also reports the
# resolved backend in every response, so it is visible at runtime too.
RUN if [ -d "${LAMBDA_TASK_ROOT}/models" ]; then \
      echo "neural feature weights present"; \
    else \
      echo "no feature weights in the build context: the classical backend will be used"; \
    fi

# Warm the OpenCV import at image build time is not possible (it happens in the
# runtime), so keep the handler's import path short and top-level: a cold start
# pays for cv2 once and then reuses it for the life of the execution environment.
CMD ["handler.handler"]
