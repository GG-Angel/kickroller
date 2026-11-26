# Upload pipeline

```text
RustFS ─ uploads.created ─▶ ingester: claim, copy to tracks/{id}, decode (PyAV),
                                      then insert Track or reject
```

The api issues upload URLs and serves reads; it never touches an upload after the PUT. The ingester owns everything from the PUT to the verdict. Both use the shared models in `kickroller.models`, and the api keeps the migrations.

Upload status: `pending` → `validating` → a `Track`, `rejected`, or `expired` if the PUT never came.

## Infra

- [x] Add a DLX policy so jobs that hit the delivery limit are kept instead of dropped (`poison` queue).
- [x] Add a bucket lifecycle rule that expires `uploads/` after a day (`infra/rustfs/lifecycle.json`).
- [x] Remove `uploads.validate` and `uploads.validated` from `infra/rabbitmq/setup.sh` and delete them from the live broker.

## Shared models (`libs/kickroller`)

- [x] Move the `Upload` and `Track` models (and `Base`) into `kickroller.models` and have the api import them.
- [x] Point the api's alembic env at the shared metadata.
- [x] Drop `UploadStatus.UPLOADED`, with a migration.

## api

- [x] Remove `handle_upload_created` and its router. Two consumers on `uploads.created` would split the events between them.

## Ingester (new app in `apps/backend/ingester`)

- [ ] FastStream consumer on `uploads.created`. Decode the event's URL-encoded keys.
- [ ] One `process(object_key)` runs the whole pipeline. The event handler and the sweep both call it.
- [ ] Claim: `UPDATE upload SET status = 'validating', validation_started_at = now(), validation_attempts = validation_attempts + 1 WHERE object_key = ? AND (status = 'pending' OR (status = 'validating' AND validation_started_at < now() - <lease>)) RETURNING ...`. No row means someone else has it: ack and exit. The lease (10 min) only has to outlast a max-size copy and decode; it lets another instance take over from one that died.
- [ ] If `validation_attempts` is over the cap (3), reject with "validation kept failing". This stops a file that crashes the ingester from looping forever.
- [ ] Copy `uploads/{id}` to `tracks/{id}` and validate the copy, since the client can still PUT to the original. The copy must be outside `uploads/`, or the bucket rule deletes it.
- [ ] Reject if the copy is over the max size. The signed PUT doesn't limit it.
- [ ] With PyAV (`av`, bundles FFmpeg) in `asyncio.to_thread`: open the file, require an allowed container (`container.format.name`) with an audio stream, decode `streams.audio[0]` summing `frame.samples` for `duration_ms`, and check the duration limits. Any `av.FFmpegError` is a rejection.
- [ ] Verdicts only apply `WHERE status = 'validating'`. If no row matches, another instance finished first: drop the result and leave `tracks/{id}` alone.
  - Pass: in one transaction, insert the `Track` (key `tracks/{id}`, `duration_ms`) and delete the upload row.
  - Fail: set `REJECTED` and `rejection_reason`, then delete `tracks/{id}`.
- [ ] Give it credentials scoped to the `upload` and `track` tables and the `uploads/` and `tracks/` prefixes.
- [ ] Add a service in `compose.yml` that depends on `rabbitmq-setup` and `api-migration`.

## Decisions

- `title` is the original filename without its extension (truncated to `TRACK_TITLE_MAX_LENGTH`). `artist` starts as null. The user can edit both later.
- Max size is 400 MiB. A 20-minute 24-bit/48 kHz stereo WAV is ~330 MiB, the largest common case at the max duration; FLAC and MP3 are far smaller. The api rejects a larger `size_bytes` at create time, and the ingester checks the real size again.
- Allowed containers: mp3, wav, flac, ogg and m4a (PyAV's mov/mp4 family). Clients play the stored file directly, so these are the formats an `<audio>` element plays everywhere. AIFF is out: only Safari plays it.
- Duration is 5 s to 20 min. That is long enough for extended mixes of a single track, and DJ sets are out of scope.
- The upload URL TTL stays at the current 15 minutes (`upload_url_ttl`). A signed PUT only has to *start* before it expires, so slow uploads of large files still finish.

## Sweep (a periodic loop in the ingester, already in TODO.md)

The event is the fast path and the database is the source of truth. Every few minutes:

- [ ] `process(key)` for `validating` uploads whose lease ran out.
- [ ] For `pending` uploads older than a few minutes, HEAD the object: if it exists, `process(key)` (its event was lost or went to `poison`); if not and `expires_at` has passed, mark `EXPIRED`.
