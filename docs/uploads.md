# Upload pipeline

```
RustFS ─ uploads.created ─▶ api: claim, copy to tracks/{id}
api ─ uploads.validate ─▶ worker: ffprobe + full decode
worker ─ uploads.validated ─▶ api: insert Track, or reject
```

The api is the only service that writes to Postgres. The worker only reads audio and returns a verdict.

## Infra

- [x] Declare `uploads.validate` and `uploads.validated` in `infra/rabbitmq/setup.sh` (quorum, published through the default exchange).
- [x] Add a DLX policy so jobs that hit the delivery limit are kept instead of dropped (`poison` queue).
- [x] Add a bucket lifecycle rule that expires `uploads/` after a day (`infra/rustfs/lifecycle.json`).

## api: `handle_upload_created`

- [ ] Claim: `UPDATE upload SET status = 'validating', validation_started_at = now(), validation_attempts = validation_attempts + 1 WHERE id = ? AND status = 'pending' RETURNING ...`. If no row comes back, ack and exit. This handles re-PUTs and redeliveries.
- [ ] Copy `uploads/{id}` to `tracks/{id}` (`copy_async`). It must be outside `uploads/`, or the bucket rule fires again.
- [ ] Publish `{upload_id, key}` to `uploads.validate`.
- [ ] Drop `UploadStatus.UPLOADED` (left over from the confirm-endpoint design), with a migration.

## Worker (new app in `apps/backend/`)

- [ ] FastStream consumer on `uploads.validate`, with `ffmpeg` in its image.
- [ ] Decide how it reads the copy: a presigned GET in the job (no credentials, but the TTL must outlast queue wait) or read-only RustFS credentials.
- [ ] Probe with `ffprobe -show_format -show_streams -of json`. Require exactly one audio stream and an allowed format (mp3, wav, flac, ogg, m4a), and check the size against the declared size and the maximum.
- [ ] Fully decode with `ffmpeg -v error -i <file> -f null -` to catch truncated or corrupt files and get an accurate `duration_ms`. Check the minimum and maximum duration.
- [ ] Run both as subprocesses with a timeout.
- [ ] Publish `{upload_id, ok, reason, duration_ms}` to `uploads.validated`.
- [ ] Add a service in `compose.yml` that depends on `rabbitmq-setup`.

## api: `handle_upload_validated`

- [ ] Ignore the message unless the upload is still `validating`.
- [ ] Pass: in one transaction, insert the `Track` (key `tracks/{id}`, `duration_ms`) and delete or close out the upload row.
- [ ] Fail: set `REJECTED` and `rejection_reason`, then delete `tracks/{id}`.

## Open questions

- Where do `title` and `artist` come from: the filename, ffprobe tags, or the user afterwards?
- What are the size and duration limits?
- What is the upload URL TTL?

## Sweeper (already in TODO.md)

- [ ] Re-queue or reject uploads stuck in `validating` (stale `validation_started_at`), with a cap on `validation_attempts`.
- [ ] Mark `pending` uploads past `expires_at` as `EXPIRED`.
