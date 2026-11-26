# Upload pipeline

```text
RustFS ─ uploads.created ─▶ ingester: claim, copy to tracks/{id}, ffprobe + full decode,
                                      then insert Track or reject
```

The api issues upload URLs and serves reads; it never touches an upload after the PUT. The ingester owns everything from the PUT to the verdict. Both use the shared models in `kickroller.models`, and the api keeps the migrations.

Upload status: `pending` → `validating` → a `Track`, `rejected`, or `expired` if the PUT never came.

## Infra

- [x] Add a DLX policy so jobs that hit the delivery limit are kept instead of dropped (`poison` queue).
- [x] Add a bucket lifecycle rule that expires `uploads/` after a day (`infra/rustfs/lifecycle.json`).
- [x] Remove `uploads.validate` and `uploads.validated` from `infra/rabbitmq/setup.sh` and delete them from the live broker.

## Shared models (`libs/kickroller`)

- [ ] Move the `Upload` and `Track` models (and `Base`) into `kickroller.models` and have the api import them.
- [ ] Point the api's alembic env at the shared metadata.
- [ ] Drop `UploadStatus.UPLOADED`, with a migration.

## api

- [x] Remove `handle_upload_created` and its router. Two consumers on `uploads.created` would split the events between them.

## Ingester (new app in `apps/backend/ingester`)

- [ ] FastStream consumer on `uploads.created`, with `ffmpeg` in its image. Decode the event's URL-encoded keys.
- [ ] Put the whole pipeline (claim, copy, validate, verdict) in one `process(object_key)`. The event handler and the sweep both call it.
- [ ] Claim with a lease: `UPDATE upload SET status = 'validating', validation_started_at = now(), validation_attempts = validation_attempts + 1 WHERE object_key = ? AND (status = 'pending' OR (status = 'validating' AND validation_started_at < now() - <lease>)) RETURNING ...`. If no row comes back, ack and exit. This handles re-PUTs, redeliveries, other ingester instances, and takes over from an instance that died mid-validation once its lease runs out.
- [ ] Set the lease longer than the copy plus the ffprobe and ffmpeg timeouts.
- [ ] Keep the `validation_attempts` the claim returns as a fencing token. An instance that lost its connection but kept running must not write a verdict over a newer claim.
- [ ] Cap attempts: if the claim returns `validation_attempts` over the cap, reject with "validation kept failing" instead of validating. This stops files that crash the ingester every time.
- [ ] Copy `uploads/{id}` to `tracks/{id}` (`copy_async`) and validate only the copy, since the client can still PUT to the original. The copy must be outside `uploads/`, or the bucket rule fires again.
- [ ] Probe with `ffprobe -show_format -show_streams -of json`. Require exactly one audio stream and an allowed format (mp3, wav, flac, ogg, m4a), and check the size against the declared size and the maximum.
- [ ] Fully decode with `ffmpeg -v error -i <file> -f null -` to catch truncated or corrupt files and get an accurate `duration_ms`. Check the minimum and maximum duration.
- [ ] Run both as subprocesses with a timeout.
- [ ] Fence the verdict: every verdict write checks `WHERE status = 'validating' AND validation_attempts = <token>`. If no row matches, a newer claim owns the upload, so drop the result and leave `tracks/{id}` alone.
- [ ] Pass: in one transaction, insert the `Track` (key `tracks/{id}`, `duration_ms`) and delete or close out the upload row.
- [ ] Fail: set `REJECTED` and `rejection_reason`, then delete `tracks/{id}`.
- [ ] Give it database credentials limited to `upload` and `track`, and RustFS credentials that can read `uploads/` and write `tracks/`.
- [ ] Add a service in `compose.yml` that depends on `rabbitmq-setup` and `api-migration`.

## Decisions

- `title` is the original filename without its extension (truncated to `TRACK_TITLE_MAX_LENGTH`). `artist` starts as null. The user can edit both later.
- Max size is 400 MiB. A 20-minute 24-bit/48 kHz stereo WAV is ~330 MiB, the largest common case at the max duration; FLAC and MP3 are far smaller. The api rejects a larger `size_bytes` at create time, and the ingester checks the real size again.
- Duration is 5 s to 20 min. That is long enough for extended mixes of a single track, and DJ sets are out of scope.
- The upload URL TTL stays at the current 15 minutes (`upload_url_ttl`). A signed PUT only has to *start* before it expires, so slow uploads of large files still finish.

## Sweep (a periodic loop in the ingester, already in TODO.md)

The event is the fast path and the database is the source of truth. Every few minutes:

- [ ] Call `process(key)` for `validating` uploads whose lease has run out (a redelivery inside the lease was acked and skipped, so nothing else retries them).
- [ ] Call `process(key)` for `pending` uploads older than a few minutes whose object exists (HEAD it). Their event was lost or landed in `poison`.
- [ ] Mark `pending` uploads past `expires_at` with no object as `EXPIRED`.
