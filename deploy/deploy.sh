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
#   ./deploy/deploy.sh              # deploy for real
#   ./deploy/deploy.sh --dry-run    # print what would be created, touch nothing
#
# Everything it creates is tagged Project=strata so teardown.sh can find it.

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg (try --dry-run)" >&2; exit 2 ;;
  esac
done

# In dry-run nothing is created, so the account id is not needed and asking for
# it would make the mode useless: it is the one command a reviewer can run
# without an account to see exactly what a deployment would do.
if [ "$DRY_RUN" = "1" ]; then
  ACCOUNT="${STRATA_ACCOUNT:-<account>}"
else
  ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
fi

REGION="${AWS_REGION:-${STRATA_AWS_REGION:-us-east-1}}"
PREFIX="${STRATA_PREFIX:-strata}"
FUNCTION="${PREFIX}-vision"
ROLE_NAME="${PREFIX}-lambda-role"
BUCKET="${STRATA_BUCKET:-${PREFIX}-sessions-${ACCOUNT}}"
TABLE="${STRATA_TABLE:-${PREFIX}-ledger}"
REPO="${PREFIX}-vision"
TAG="${STRATA_TAG:-$(date +%Y%m%d-%H%M%S)}"
# A public function URL is an unauthenticated compute endpoint. Reserving
# concurrency bounds what one runaway client can bill: at 3008 MB x 5
# concurrent executions the worst case is bounded, not open-ended.
RESERVED_CONCURRENCY="${STRATA_RESERVED_CONCURRENCY:-5}"

REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '    [dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

say "account ${ACCOUNT} · region ${REGION}$([ "$DRY_RUN" = "1" ] && printf ' · dry run')"

# ---------------------------------------------------------------- S3 ---------
if [ "$DRY_RUN" = "1" ] || ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "    creating bucket: ${BUCKET}"
  if [ "$REGION" = "us-east-1" ]; then
    run aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    run aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
  run aws s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
  run aws s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
else
  echo "    bucket exists: ${BUCKET}"
fi
[ "$DRY_RUN" = "1" ] || aws s3api put-bucket-tagging --bucket "$BUCKET" \
  --tagging "TagSet=[{Key=Project,Value=${PREFIX}}]" 2>/dev/null || true

# ------------------------------------------------------------ DynamoDB ------
# The key schema is the storage contract that service/store.py writes against:
# pk is the parent of a record key, sk its leaf. Changing either side without
# the other makes the first put_item in production a ValidationException, so
# tests/test_store_aws.py asserts this block and that module agree.
if [ "$DRY_RUN" = "1" ] || ! aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" >/dev/null 2>&1; then
  echo "    creating table: ${TABLE}"
  run aws dynamodb create-table --table-name "$TABLE" --region "$REGION" \
    --attribute-definitions AttributeName=pk,AttributeType=S \
                            AttributeName=sk,AttributeType=S \
    --key-schema AttributeName=pk,KeyType=HASH \
                 AttributeName=sk,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --tags Key=Project,Value="$PREFIX"
  [ "$DRY_RUN" = "1" ] || aws dynamodb wait table-exists --table-name "$TABLE" --region "$REGION"
else
  echo "    table exists: ${TABLE}"
fi

# ----------------------------------------------------------------- ECR ------
if [ "$DRY_RUN" = "1" ] || ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  echo "    creating repo: ${REPO}"
  run aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true \
    --tags Key=Project,Value="$PREFIX"
else
  echo "    repo exists: ${REPO}"
fi

if [ "$DRY_RUN" = "1" ]; then
  printf '    [dry-run] docker buildx build --platform linux/arm64 -f deploy/Dockerfile -t %s --push .\n' "$IMAGE"
else
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
  say "building arm64 image ${IMAGE}"
  docker buildx build --platform linux/arm64 \
    -f deploy/Dockerfile -t "$IMAGE" --push .
fi

# ------------------------------------------------------------- IAM role -----
if [ "$DRY_RUN" = "1" ] || ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "    creating role: ${ROLE_NAME}"
  run aws iam create-role --role-name "$ROLE_NAME" --tags Key=Project,Value="$PREFIX" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"}]}' >/dev/null
  run aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
else
  echo "    role exists: ${ROLE_NAME}"
fi

# Scoped to the two resources this function actually touches. GetItem is listed
# because the store reads records back, not only writes them.
run aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name strata-data-access \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],
       \"Resource\":\"arn:aws:s3:::${BUCKET}/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"dynamodb:PutItem\",\"dynamodb:GetItem\",\"dynamodb:Query\"],
       \"Resource\":\"arn:aws:dynamodb:${REGION}:${ACCOUNT}:table/${TABLE}\"},
      {\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:PutMetricData\"],\"Resource\":\"*\"}
    ]}"

if [ "$DRY_RUN" = "1" ]; then
  ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
else
  ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"
fi

ENV_VARS="Variables={STRATA_S3_BUCKET=${BUCKET},STRATA_DDB_TABLE=${TABLE},STRATA_AWS_REGION=${REGION},STRATA_BACKEND=auto}"

# ------------------------------------------------------------- Lambda -------
if [ "$DRY_RUN" = "0" ] && aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  say "updating function"
  run aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" \
    --image-uri "$IMAGE"
  run aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
  run aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
    --timeout 120 --memory-size 3008 \
    --environment "$ENV_VARS"
else
  say "creating function"
  # A role that was just created is not assumable immediately, and Lambda says
  # so with a ValidationError. Polling the error we actually expect beats a
  # fixed sleep that is simultaneously too short and too long.
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [ "$DRY_RUN" = "1" ]; then
      run aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
        --package-type Image --code ImageUri="$IMAGE" \
        --role "$ROLE_ARN" --architectures arm64 \
        --timeout 120 --memory-size 3008 \
        --environment "$ENV_VARS" \
        --tags Project="$PREFIX"
      break
    fi
    if ERR="$(aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
        --package-type Image --code ImageUri="$IMAGE" \
        --role "$ROLE_ARN" --architectures arm64 \
        --timeout 120 --memory-size 3008 \
        --environment "$ENV_VARS" \
        --tags Project="$PREFIX" 2>&1 >/dev/null)"; then
      break
    fi
    if ! printf '%s' "$ERR" | grep -q "not authorized to perform: lambda:CreateFunction on role\|The role defined for the function cannot be assumed by Lambda"; then
      printf '%s\n' "$ERR" >&2
      exit 1
    fi
    echo "    role not assumable yet (attempt ${attempt}/10)"
    sleep 5
  done
  [ "$DRY_RUN" = "1" ] || aws lambda wait function-active --function-name "$FUNCTION" --region "$REGION"
fi

run aws lambda put-function-concurrency --function-name "$FUNCTION" --region "$REGION" \
  --reserved-concurrent-executions "$RESERVED_CONCURRENCY"

# --------------------------------------------------------- Function URL -----
# Auth stays NONE because the smoke test and the judge demo hit it directly;
# the reserved concurrency above is what bounds the cost of an open endpoint.
# A deployment that puts this behind a real front door should switch to
# AWS_IAM and drop the public permission statement.
if [ "$DRY_RUN" = "0" ] && URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" --query FunctionUrl --output text 2>/dev/null)"; then
  echo "    function url exists"
else
  URL="$( [ "$DRY_RUN" = "1" ] && printf 'https://<dry-run>.lambda-url.%s.on.aws/' "$FUNCTION" || \
    aws lambda create-function-url-config --function-name "$FUNCTION" \
      --region "$REGION" --auth-type NONE --query FunctionUrl --output text )"
  run aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl --principal '*' \
    --function-url-auth-type NONE
fi

cat <<EOF

────────────────────────────────────────────────────────────
  function   ${FUNCTION}
  region     ${REGION}
  bucket     ${BUCKET}
  table      ${TABLE}
  image      ${IMAGE}
  endpoint   ${URL}

  Next: prove it ran, end to end, with real images and a real ledger write:

    python deploy/verify.py --bucket ${BUCKET} --function ${FUNCTION} \\
      --region ${REGION} --out deploy/verify-report.json
────────────────────────────────────────────────────────────
EOF
