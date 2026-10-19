#!/usr/bin/env bash
# Run the benchmark on AWS Graviton4 and collect the result.
#
# Fully automated: the instance is launched, benchmarks itself from user-data,
# uploads its JSON to S3 and shuts down. Nothing is left running, and no SSH key
# has to exist.
#
#   COOL_AMI_ID=ami-xxxxxxxx ./deploy/benchmark/run_on_graviton.sh
#
# COOL_AMI_ID comes from the Cloud-Optimized OpenCV listing on AWS Marketplace:
# subscribe to the product, then launch from the console once to discover the
# AMI id for your region (or use `aws ec2 describe-images --owners <seller>`).
#
# The instance is sized to the task and terminated at the end. Expected cost for
# a default run: well under a dollar of compute, plus the COOL software rate
# (which the Marketplace free trial covers for seven days).

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${STRATA_BENCH_INSTANCE:-c8g.xlarge}"
AMI="${COOL_AMI_ID:?set COOL_AMI_ID to the Cloud-Optimized OpenCV AMI for ${REGION}}"
PREFIX="${STRATA_PREFIX:-strata}"
BUCKET="${STRATA_BUCKET:?set STRATA_BUCKET to a bucket you own in ${REGION}}"
ROLE_NAME="${PREFIX}-bench-role"
PROFILE_NAME="${PREFIX}-bench-profile"
REPEAT="${STRATA_BENCH_REPEAT:-3}"
TARBALL="strata-bench.tar.gz"
KEY="bench/result-arm.json"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- package ---
say "packaging the benchmark"
rm -f "/tmp/${TARBALL}"
INCLUDE=(vision deploy/benchmark)
if [ "${STRATA_BENCH_WITH_MODELS:-1}" = "1" ] && [ -d vision/models ]; then
  INCLUDE+=(vision/models)
  echo "    including ONNX weights (neural backend available on the instance)"
else
  echo "    excluding ONNX weights (classical backend only)"
fi
tar czf "/tmp/${TARBALL}" "${INCLUDE[@]}"
aws s3 cp "/tmp/${TARBALL}" "s3://${BUCKET}/bench/${TARBALL}" --region "$REGION" >/dev/null

# ------------------------------------------------------------ instance role -
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  say "creating instance role"
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
        "Principal":{"Service":"ec2.amazonaws.com"},
        "Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name strata-bench-write \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",
      \"Action\":[\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::${BUCKET}/bench/*\"}]}"
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" \
    --role-name "$ROLE_NAME" >/dev/null
  sleep 10
fi
PROFILE_ARN="$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" \
  --query InstanceProfile.Arn --output text)"

# --------------------------------------------------------------- user data --
read -r -d '' USER_DATA <<EOS || true
#!/bin/bash
set -x
exec > /var/log/strata-bench.log 2>&1

# The COOL AMI ships preconfigured virtualenvs; use the newest available so the
# measured cv2 is the COOL build and not a system copy.
VENV=""
for v in /opt/cool/venvs/python_3.12 /opt/cool/venvs/python_3.11 /opt/cool/venvs/python_3.10; do
  if [ -d "\$v" ]; then VENV="\$v"; break; fi
done
if [ -z "\$VENV" ]; then echo "no COOL venv found"; shutdown -h now; exit 1; fi
source "\$VENV/bin/activate"

python -c "import cv2; print('cv2', cv2.__version__, cv2.__file__)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy || true

cd /opt
aws s3 cp s3://${BUCKET}/bench/${TARBALL} /opt/strata-bench.tar.gz --region ${REGION}
tar xzf /opt/strata-bench.tar.gz -C /opt

export STRATA_COOL_VERSION="COOL 3.1"
cd /opt
python deploy/benchmark/bench_core.py \
  --json /tmp/result.json \
  --repeat ${REPEAT} \
  --instance ${INSTANCE_TYPE} \
  --backend auto

aws s3 cp /tmp/result.json s3://${BUCKET}/${KEY} --region ${REGION}
shutdown -h now
EOS

# ---------------------------------------------------------------- launch ----
say "launching ${INSTANCE_TYPE} from ${AMI}"
INSTANCE_ID="$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --iam-instance-profile "Arn=${PROFILE_ARN}" \
  --user-data "$USER_DATA" \
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=${PREFIX}},{Key=Name,Value=${PREFIX}-bench}]" \
  --query 'Instances[0].InstanceId' --output text)"

echo "    ${INSTANCE_ID}"

cleanup() {
  say "terminating ${INSTANCE_ID}"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null || true
}
trap cleanup EXIT

# ------------------------------------------------------------------ wait ----
say "waiting for the result (up to 20 minutes)"
for i in $(seq 1 120); do
  if aws s3api head-object --bucket "$BUCKET" --key "$KEY" --region "$REGION" >/dev/null 2>&1; then
    echo "    result available after ~$((i * 10))s"
    break
  fi
  sleep 10
done

if ! aws s3api head-object --bucket "$BUCKET" --key "$KEY" --region "$REGION" >/dev/null 2>&1; then
  echo "!! no result after 20 minutes - check /var/log/strata-bench.log on the instance"
  exit 1
fi

OUT="deploy/benchmark/results/arm-${INSTANCE_TYPE}.json"
mkdir -p "$(dirname "$OUT")"
aws s3 cp "s3://${BUCKET}/${KEY}" "$OUT" --region "$REGION" >/dev/null
echo "    wrote ${OUT}"
