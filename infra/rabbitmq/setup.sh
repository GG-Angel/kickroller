#!/bin/sh
# Declares the queues consumers read from. Safe to rerun.
set -eu

# Events from RustFS
rabbitmqadmin declare queue --name uploads.created --type quorum --durable true
rabbitmqadmin declare binding --source amq.direct --destination-type queue \
  --destination uploads.created --routing-key storage.events

# Upload validation jobs and their results
rabbitmqadmin declare queue --name uploads.validate --type quorum --durable true
rabbitmqadmin declare queue --name uploads.validated --type quorum --durable true

# Failed messages
rabbitmqadmin declare queue --name poison --type quorum --durable true
rabbitmqadmin declare policy --name uploads-poison --pattern '^uploads\.' \
  --apply-to quorum_queues \
  --definition '{"dead-letter-exchange": "", "dead-letter-routing-key": "poison"}'
