#!/bin/sh
# Creates the bucket and its event rules. Safe to rerun.
set -eu

bucket=kickroller

aws s3api head-bucket --bucket "$bucket" 2>/dev/null \
  || aws s3api create-bucket --bucket "$bucket"

aws s3api put-bucket-notification-configuration --bucket "$bucket" \
  --notification-configuration "file://$(dirname "$0")/notification.json"
