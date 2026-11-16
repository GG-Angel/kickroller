# api

FastAPI service. Run the commands below from this directory, with `.env` copied from `.env.example`.

## Run

```sh
docker compose up -d db rustfs   # Postgres and object storage
uv run poe migrate
uv run api                       # http://localhost:8000
```

The API checks the `kickroller` bucket at startup. Create it once (obstore can't create buckets):

```sh
curl --aws-sigv4 "aws:amz:us-east-1:s3" --user rustfsadmin:rustfsadmin \
  -X PUT http://localhost:9000/kickroller
```

## Tasks

| Command                     | What                                     |
| --------------------------- | ---------------------------------------- |
| `uv run poe test`           | Run the tests                            |
| `uv run poe fix`            | Lint with fixes, then format             |
| `uv run poe migrate`        | Apply migrations                         |
| `uv run poe revision "msg"` | Autogenerate a migration from the models |

New revisions are fixed and formatted automatically.
