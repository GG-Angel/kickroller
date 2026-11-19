#!/bin/sh
# Creates the bucket, its event rules and its lifecycle rules. Safe to rerun.
set -eu

bucket=kickroller

aws s3api head-bucket --bucket "$bucket" 2>/dev/null \
  || aws s3api create-bucket --bucket "$bucket"

aws s3api put-bucket-notification-configuration --bucket "$bucket" \
  --notification-configuration "file://$(dirname "$0")/notification.json"

# Raw uploads are only needed until they're copied out, so expire leftovers.
aws s3api put-bucket-lifecycle-configuration --bucket "$bucket" \
  --lifecycle-configuration "file://$(dirname "$0")/lifecycle.json"
