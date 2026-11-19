#!/bin/sh
# Declares the queues consumers read from. Safe to rerun.
set -eu

# Events from RustFS
rabbitmqadmin declare queue --name uploads.created --type quorum --durable true
rabbitmqadmin declare binding --source amq.direct --destination-type queue \
  --destination uploads.created --routing-key storage.events

# Validation jobs and their results, published by us to the default exchange
rabbitmqadmin declare queue --name uploads.validate --type quorum --durable true
rabbitmqadmin declare queue --name uploads.validated --type quorum --durable true
