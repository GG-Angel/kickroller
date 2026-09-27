# analyzer

Detects kick onsets in a rawstyle track and writes their times in seconds.

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (it decodes any audio or video format, including MP3, M4A and MOV).

## Usage

```sh
uv run analyzer "track.m4a"                         # CSV to stdout, one time per line
uv run analyzer "track.m4a" -o kicks.csv            # CSV file (import into Sonic Visualiser as time instants)
uv run analyzer "track.m4a" -f json -o kicks.json   # {"file": "track.m4a", "kicks": [...]}
uv run analyzer "track.m4a" --sonify check.wav      # also write the track with a click at each kick
uv run analyzer "track.m4a" -t 1.2                  # change the peak threshold
```

## Method

1. Decode with ffmpeg to 44.1 kHz, mid channel (L+R)/2, loudness-normalize to -14 LUFS.
2. Candidates: SuperFlux onset strength in the click band (2-6 kHz, 12 log bands per octave, 46 ms Hann window, 10 ms hop), then local maxima within +/-20 ms.
3. Low-band check (30-300 Hz), comparing 30 ms after the attack with 30 ms before. A candidate is a kick if one of these is true:
   - Its click strength is at least the threshold, and it starts a new kick body: the low-band centroid jumps up by at least 12 Hz (the kick's pitch drop starts high), or the low-band energy rises by at least 6 dB (a kick after a gap). The low band must also have energy after the attack.
   - Its click is weak (at least 0.5), but the low band restarts at full level after a gap: a rise of at least 12 dB to within 10 dB of the track's loud level.

   This rejects attacks inside a kick tail (tail gating, screeches, claps).
4. Remove kicks closer than 40 ms to a stronger kick.

Known limits: kicks closer than about 60 ms (dense rolls) can fail the low-band check. Kicks that cut a tail with a weak click (below the threshold) are missed. The thresholds are tuned by ear on two tracks only.

Audio files must never be committed (see `.gitignore`).
