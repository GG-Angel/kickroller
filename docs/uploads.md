# Upload pipeline

```
RustFS ─ uploads.created ─▶ worker: claim, copy to tracks/{id}, ffprobe + full decode,
                                    then insert Track or reject
```

The api issues upload URLs and serves reads; it never touches an upload after the PUT. The worker owns everything from the PUT to the verdict. Both use the shared models in `libs/`, and the api keeps the migrations.

Upload status: `pending` → `validating` → a `Track`, `rejected`, or `expired` if the PUT never came.

## Infra

- [x] Add a DLX policy so jobs that hit the delivery limit are kept instead of dropped (`poison` queue).
- [x] Add a bucket lifecycle rule that expires `uploads/` after a day (`infra/rustfs/lifecycle.json`).
- [ ] Remove `uploads.validate` and `uploads.validated` from `infra/rabbitmq/setup.sh` and delete them from the live broker.

## Shared models (`libs/`)

- [ ] Move the `Upload` and `Track` models (and `Base`) into a package in `libs/` and have the api import them.
- [ ] Point the api's alembic env at the shared metadata.
- [ ] Drop `UploadStatus.UPLOADED`, with a migration.

## api

- [ ] Remove `handle_upload_created` and its router. Two consumers on `uploads.created` would split the events between them.

## Worker (new app in `apps/backend/`)

- [ ] FastStream consumer on `uploads.created`, with `ffmpeg` in its image. Decode the event's URL-encoded keys.
- [ ] Claim: `UPDATE upload SET status = 'validating', validation_started_at = now(), validation_attempts = validation_attempts + 1 WHERE object_key = ? AND status = 'pending' RETURNING ...`. If no row comes back, ack and exit. This handles re-PUTs, redeliveries and other workers.
- [ ] Copy `uploads/{id}` to `tracks/{id}` (`copy_async`) and validate only the copy, since the client can still PUT to the original. The copy must be outside `uploads/`, or the bucket rule fires again.
- [ ] Probe with `ffprobe -show_format -show_streams -of json`. Require exactly one audio stream and an allowed format (mp3, wav, flac, ogg, m4a), and check the size against the declared size and the maximum.
- [ ] Fully decode with `ffmpeg -v error -i <file> -f null -` to catch truncated or corrupt files and get an accurate `duration_ms`. Check the minimum and maximum duration.
- [ ] Run both as subprocesses with a timeout.
- [ ] Pass: in one transaction, insert the `Track` (key `tracks/{id}`, `duration_ms`) and delete or close out the upload row.
- [ ] Fail: set `REJECTED` and `rejection_reason`, then delete `tracks/{id}`.
- [ ] Give it database credentials limited to `upload` and `track`, and RustFS credentials that can read `uploads/` and write `tracks/`.
- [ ] Add a service in `compose.yml` that depends on `rabbitmq-setup` and `api-migration`.

## Open questions

- Where do `title` and `artist` come from: the filename, ffprobe tags, or the user afterwards?
- What are the size and duration limits?
- What is the upload URL TTL?

## Sweeper (already in TODO.md)

- [ ] Release or reject uploads stuck in `validating` (stale `validation_started_at`, so the worker died holding the claim), with a cap on `validation_attempts`. Stale must be longer than the validation timeout.
- [ ] Handle `pending` uploads whose object exists but whose event was lost or landed in `poison`.
- [ ] Mark `pending` uploads past `expires_at` as `EXPIRED`.
