#!/bin/sh
# Declares the queues consumers read from. Safe to rerun.
set -eu

# Events from RustFS
rabbitmqadmin declare queue --name storage.events --type quorum --durable true
rabbitmqadmin declare binding --source amq.direct --destination-type queue \
  --destination storage.events --routing-key storage.events
