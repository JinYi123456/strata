#!/usr/bin/env bash
# Deploy the STRATA vision workload to AWS Lambda on Graviton (arm64).
#
# Creates, idempotently:
#   - an S3 bucket for session objects
#   - a DynamoDB table for the site ledger
#   - an ECR repository holding an arm64 container image
#   - a Lambda function built from that image, on arm64
#   - a public Function URL in front of it
#
# Requires: awscli v2, docker with buildx, and credentials for the target account.
#
#   ./deploy/deploy.sh
#
# Everything it creates is tagged Project=strata so teardown.sh can find it.

set -euo pipefail

REGION="${AWS_REGION:-${STRATA_AWS_REGION:-us-east-1}}"
PREFIX="${STRATA_PREFIX:-strata}"
FUNCTION="${PREFIX}-vision"
ROLE_NAME="${PREFIX}-lambda-role"
BUCKET="${STRATA_BUCKET:-${PREFIX}-sessions-$(aws sts get-caller-identity --query Account --output text)}"
TABLE="${STRATA_TABLE:-${PREFIX}-ledger}"
REPO="${PREFIX}-vision"
TAG="${STRATA_TAG:-$(date +%Y%m%d-%H%M%S)}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "account ${ACCOUNT} · region ${REGION}"

# ---------------------------------------------------------------- S3 ---------
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "    bucket exists: ${BUCKET}"
else
  echo "    creating bucket: ${BUCKET}"
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  aws s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
fi
aws s3api put-bucket-tagging --bucket "$BUCKET" \
  --tagging "TagSet=[{Key=Project,Value=${PREFIX}}]" 2>/dev/null || true

# ------------------------------------------------------------ DynamoDB ------
if aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" >/dev/null 2>&1; then
  echo "    table exists: ${TABLE}"
else
  echo "    creating table: ${TABLE}"
  aws dynamodb create-table --table-name "$TABLE" --region "$REGION" \
    --attribute-definitions AttributeName=site_id,AttributeType=S \
                            AttributeName=session_id,AttributeType=S \
    --key-schema AttributeName=site_id,KeyType=HASH \
                 AttributeName=session_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --tags Key=Project,Value="$PREFIX" >/dev/null
  aws dynamodb wait table-exists --table-name "$TABLE" --region "$REGION"
fi

# ----------------------------------------------------------------- ECR ------
if aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  echo "    repo exists: ${REPO}"
else
  echo "    creating repo: ${REPO}"
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true \
    --tags Key=Project,Value="$PREFIX" >/dev/null
fi

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

say "building arm64 image ${IMAGE}"
docker buildx build --platform linux/arm64 \
  -f deploy/Dockerfile -t "$IMAGE" --push .

# ------------------------------------------------------------- IAM role -----
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "    role exists: ${ROLE_NAME}"
else
  echo "    creating role: ${ROLE_NAME}"
  aws iam create-role --role-name "$ROLE_NAME" --tags Key=Project,Value="$PREFIX" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi

# Scoped to the two resources this function actually touches.
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name strata-data-access \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],
       \"Resource\":\"arn:aws:s3:::${BUCKET}/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"dynamodb:PutItem\",\"dynamodb:Query\"],
       \"Resource\":\"arn:aws:dynamodb:${REGION}:${ACCOUNT}:table/${TABLE}\"},
      {\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:PutMetricData\"],\"Resource\":\"*\"}
    ]}"

ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

ENV_VARS="Variables={STRATA_S3_BUCKET=${BUCKET},STRATA_DDB_TABLE=${TABLE},STRATA_AWS_REGION=${REGION},STRATA_BACKEND=auto}"

# ------------------------------------------------------------- Lambda -------
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  say "updating function"
  aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" \
    --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
    --timeout 120 --memory-size 3008 \
    --environment "$ENV_VARS" >/dev/null
else
  say "creating function"
  # Give IAM a moment to make the new role assumable.
  sleep 12
  aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
    --package-type Image --code ImageUri="$IMAGE" \
    --role "$ROLE_ARN" --architectures arm64 \
    --timeout 120 --memory-size 3008 \
    --environment "$ENV_VARS" \
    --tags Project="$PREFIX" >/dev/null
  aws lambda wait function-active --function-name "$FUNCTION" --region "$REGION"
fi

# --------------------------------------------------------- Function URL -----
if URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" --query FunctionUrl --output text 2>/dev/null)"; then
  echo "    function url exists"
else
  URL="$(aws lambda create-function-url-config --function-name "$FUNCTION" \
    --region "$REGION" --auth-type NONE --query FunctionUrl --output text)"
  aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl --principal '*' \
    --function-url-auth-type NONE >/dev/null
fi

cat <<EOF

────────────────────────────────────────────────────────────
  function   ${FUNCTION}
  region     ${REGION}
  bucket     ${BUCKET}
  table      ${TABLE}
  image      ${IMAGE}
  endpoint   ${URL}

  smoke test:

  aws lambda invoke --function-name ${FUNCTION} --region ${REGION} \\
    --cli-binary-format raw-in-base64-out \\
    --payload '{"site_id":"demo","session_id":"s1","baseline":["demo/a.jpg"],"rescan":["demo/b.jpg"]}' \\
    /tmp/out.json && cat /tmp/out.json
────────────────────────────────────────────────────────────
EOF
