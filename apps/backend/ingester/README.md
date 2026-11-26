# ingester

Turns uploaded audio into tracks: claims each upload, validates it with ffmpeg and records the verdict. See [docs/uploads.md](../../../docs/uploads.md). Run the commands below from this directory.

## Run

```sh
uv run ingester
```

## Tasks

| Command           | What                         |
| ----------------- | ---------------------------- |
| `uv run poe test` | Run the tests                |
| `uv run poe fix`  | Lint with fixes, then format |
