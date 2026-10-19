#!/usr/bin/env bash
# Remove everything deploy.sh created, and nothing else.
#
#   ./deploy/teardown.sh            # asks once
#   ./deploy/teardown.sh --yes      # no prompt
#
# Deliberately narrow: it deletes by exact name, never by tag or prefix, so a
# mistyped environment variable cannot take unrelated resources with it.

set -euo pipefail

REGION="${AWS_REGION:-${STRATA_AWS_REGION:-us-east-1}}"
PREFIX="${STRATA_PREFIX:-strata}"
FUNCTION="${PREFIX}-vision"
ROLE_NAME="${PREFIX}-lambda-role"
BUCKET="${STRATA_BUCKET:-${PREFIX}-sessions-$(aws sts get-caller-identity --query Account --output text)}"
TABLE="${STRATA_TABLE:-${PREFIX}-ledger}"
REPO="${PREFIX}-vision"

if [ "${1:-}" != "--yes" ]; then
  cat <<EOF
This will delete, in region ${REGION}:
  lambda   ${FUNCTION}
  role     ${ROLE_NAME}
  bucket   ${BUCKET}   (and every object in it)
  table    ${TABLE}
  repo     ${REPO}     (and every image in it)
EOF
  read -r -p "Type the bucket name to confirm: " answer
  [ "$answer" = "$BUCKET" ] || { echo "aborted"; exit 1; }
fi

say() { printf '\n==> %s\n' "$*"; }

say "function url"
aws lambda delete-function-url-config --function-name "$FUNCTION" --region "$REGION" 2>/dev/null || true

say "lambda"
aws lambda delete-function --function-name "$FUNCTION" --region "$REGION" 2>/dev/null || true

say "role"
for p in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
  aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$p" || true
done
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name strata-data-access 2>/dev/null || true
aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || true

say "bucket"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3 rm "s3://${BUCKET}" --recursive --quiet || true
  aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || true
fi

say "table"
aws dynamodb delete-table --table-name "$TABLE" --region "$REGION" 2>/dev/null || true

say "repository"
aws ecr delete-repository --repository-name "$REPO" --region "$REGION" --force 2>/dev/null || true

say "done — nothing from ${PREFIX} remains in ${REGION}"
