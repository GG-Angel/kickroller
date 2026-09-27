# analyzer

Detects kick onsets in a rawstyle track and writes their times in seconds.

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (it decodes any audio or video format, including MP3, M4A and MOV).

## Usage

```sh
uv run analyzer "track.m4a"                         # CSV to stdout, one time per line
uv run analyzer "track.m4a" -o kicks.csv            # CSV file (import into Sonic Visualiser as time instants)
uv run analyzer "track.m4a" -f json -o kicks.json   # {"file": "track.m4a", "kicks": [...]}
uv run analyzer "track.m4a" --sonify check.wav      # also write the track with a click at each kick
```

The first run downloads the beat_this model (about 78 MB) to the PyTorch cache.

## Method

1. Decode with ffmpeg to 44.1 kHz, mid channel (L+R)/2.
2. Beat grid: track the beats with [beat_this](https://github.com/CPJKU/beat_this) (`final0` model, CPU, no DBN). Remove extra beats closer than 3/4 of the beat period, fill skipped beats, and extend the beats over the full track. The beat period is the median beat interval, moved by octaves toward 150-170 BPM.
3. Loudness-normalize to -14 LUFS. Candidates: SuperFlux onset strength in the click band (2-6 kHz, 12 log bands per octave, 46 ms Hann window, 10 ms hop), then local maxima within +/-20 ms.
4. Low-band features (30-300 Hz), comparing 30 ms after the attack with 30 ms before:
   - new kick body: the low band has energy after the attack, and its centroid jumps up by at least 12 Hz (the kick's pitch drop starts high) or its energy rises by at least 6 dB (a kick after a gap);
   - restart: a click of at least 0.5 where the low band restarts at full level after a gap (a rise of at least 12 dB, to within 10 dB of the track's loud level).
5. Check the grid positions, each one independently. The checks are stricter at finer positions:

   | Position | Window | Kick if |
   |---|---|---|
   | beat | +/-40 ms | click >= 0.9 and new kick body, or restart |
   | 1/8 | +/-30 ms | click >= 1.5 and new kick body, or restart |
   | 1/16 (1/4, 3/4 of a beat), triplet (1/3, 2/3) | +/-15 ms | click >= 1.5, low band within 20 dB of the loud level, and a centroid jump >= 12 Hz |

   First the beats are checked. Each beat then moves to its kick (beat_this gives beats in 20 ms steps), and the beats with no kick move by the median shift. Then the off-beat positions are checked between the moved beats. At each position, the strongest candidate is kept. In each beat, the straight (1/8, 1/16) or the triplet kicks are kept, whichever has the larger total click strength.
6. Remove kicks closer than 40 ms to a stronger kick. The output is the detected onset time, not the grid time.

This rejects attacks inside a kick tail (tail gating, screeches, claps) and all attacks off the grid.

Known limits: the output is only as good as the beat grid; a wrong beat phase or tempo gives wrong kicks, and kicks off the grid are lost. Kicks closer than about 60 ms can fail the low-band check. The thresholds are tuned by ear on two tracks only.

Audio files must never be committed (see `.gitignore`).
