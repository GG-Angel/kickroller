# analyzer

Detects kick onsets in a rawstyle track and writes their times in seconds, each with a confidence from 0 to 1. It can also write the beat grid times.

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (it decodes any audio or video format, including MP3, M4A and MOV).

## Usage

```sh
uv run analyzer analyze "track.m4a"                         # CSV to stdout, one "time,confidence" per line
uv run analyzer analyze "track.m4a" -o kicks.csv            # CSV file (Sonic Visualiser can import it)
uv run analyzer analyze "track.m4a" -f json -o kicks.json   # {"file": "track.m4a", "beats": [...], "kicks": [{"time": ..., "confidence": ...}]}
uv run analyzer analyze "track.m4a" -b beats.csv            # also write the beat grid, one time per line
uv run analyzer analyze "track.m4a" -c 0.5                  # only kicks with a confidence of at least 0.5 (default: 0.1)
uv run analyzer analyze "track.m4a" --sonify check.wav      # also write the track with a click at each kick (louder = more confident)
uv run analyzer analyze "track.m4a" -v                      # show the pipeline steps (-vv: also one line per candidate attack)
uv run analyzer analyze "track.m4a" -q                      # only show warnings and errors
```

The first run downloads the beat_this model (about 78 MB) to the PyTorch cache. Run `uv run analyzer analyze --help` for all options.

Logs go to standard error ([loguru](https://github.com/Delgan/loguru)), so standard output only has the result. The package is silent when you import it as a library; call `logger.enable("analyzer")` to see its logs.

## Development

```sh
uv run ruff format   # format
uv run ruff check    # lint
uv run ty check      # type check
```

## Method

1. Decode with ffmpeg to 44.1 kHz, mid channel (L+R)/2.
2. Beat grid: track the beats with [beat_this](https://github.com/CPJKU/beat_this) (`final0` model, CPU, no DBN). Remove extra beats closer than 3/4 of the beat period, fill skipped beats, and extend the beats over the full track. The beat period is the median beat interval, moved by octaves toward 150-170 BPM.
3. Loudness-normalize to -14 LUFS. Candidates: SuperFlux onset strength in the click band (2-6 kHz, 12 log bands per octave, 46 ms Hann window, 10 ms hop), then local maxima within +/-20 ms.
4. Low-band features (30-300 Hz): the level after the attack (relative to the track's loud level), the rise and the centroid jump (30 ms after the attack compared with 30 ms before), and the short-gap rise (the highest energy 10-30 ms after the attack compared with the lowest energy from the attack back to 20 ms before).
5. Confidence: each rule is a product of sigmoids, each centered on one threshold (0.5 at the threshold). The score for a grid position is the highest rule score. The click thresholds are stricter at finer positions: 0.9 at beats, 1.5 at 1/8, 1/16 and triplet positions.

   | Rule                                                             | Positions | Evidence                                                                                                                                                                                                                                  | Cap               |
   | ---------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
   | New kick body                                                    | all       | click >= position threshold, low band >= -35 dB, and centroid jump >= 12 Hz (the kick's pitch drop starts high) or rise >= 6 dB (a kick after a gap). At 1/16 and triplet positions: low band >= -20 dB and only the centroid jump counts | 1.0               |
   | Restart                                                          | beat, 1/8 | click >= 0.5 where the low band restarts at full level after a gap: rise >= 12 dB, low band >= -10 dB                                                                                                                                     | 1.0               |
   | Low band starts with the click (for example a regular drum kick) | beat, 1/8 | click >= 0.5, short-gap rise >= 8 dB, low band >= -25 dB                                                                                                                                                                                  | 0.5 beat, 0.4 1/8 |
   | Strong click only                                                | beat      | click >= 1.8                                                                                                                                                                                                                              | 0.3               |

   Clear hardstyle kicks (new kick body or restart) score near 1. Kicks with only the capped "maybe" evidence score below 0.5.

6. Grid positions, with search windows of +/-40 ms (beat), +/-30 ms (1/8) and +/-15 ms (1/16 at 1/4 and 3/4 of a beat, triplet at 1/3 and 2/3). First the beats are checked. Each beat then moves to its kick if that kick scores at least 0.5 (beat_this gives beats in 20 ms steps), and the other beats move by the median shift. Then the off-beat positions are checked between the moved beats. At each position, the highest-scoring candidate is kept. In each beat, the straight (1/8, 1/16) or the triplet kicks are kept, whichever has the higher best score.
7. Klaplong kicks: a punch with no bass (low band <= -20 dB after the click), then the bass half a beat later (+/-30 ms; low band >= -10 dB). The bass swells in between the two clicks, so neither click has a rise of its own (< 6 dB). The pair is one kick at the punch: the punch gets the confidence of the bass part if that is higher, and the confidence of the bass part is multiplied by (1 - the pair score).
8. Drop kicks below the minimum confidence (default 0.1), then remove kicks closer than 40 ms to a higher-scoring kick. The output is the detected onset time, not the grid time.
9. Beat output: the moved beats from step 6 that are inside the track. A beat with a confident kick is at the kick onset; the other beats are the beat_this beats moved by the median shift.

This rejects attacks inside a kick tail (tail gating, screeches, claps) and all attacks off the grid.

Known limits: the output is only as good as the beat grid; a wrong beat phase or tempo gives wrong kicks, and kicks off the grid are lost. Kicks closer than about 60 ms can fail the low-band check. The thresholds and the confidence are tuned by ear on two tracks only (the klaplong rule on one track); the confidence ranks kicks but is not a calibrated probability.

Audio files must never be committed (see `.gitignore`).
