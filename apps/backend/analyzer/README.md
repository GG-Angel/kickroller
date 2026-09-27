# analyzer

Detects kick onsets in a rawstyle track and writes their times in seconds, each with a confidence from 0 to 1. It can also write the beat grid times.

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (it decodes any audio or video format, including MP3, M4A and MOV).

## Usage

Train the kick model once (see [Training](#training)), then analyze tracks:

```sh
uv run analyzer analyze "track.m4a"                         # CSV to stdout, one "time,confidence" per line
uv run analyzer analyze "track.m4a" -o kicks.csv            # CSV file (Sonic Visualiser can import it)
uv run analyzer analyze "track.m4a" -f json -o kicks.json   # {"file": "track.m4a", "beats": [...], "kicks": [{"time": ..., "confidence": ...}]}
uv run analyzer analyze "track.m4a" -b beats.csv            # also write the beat grid, one time per line
uv run analyzer analyze "track.m4a" -c 0.5                  # only kicks with a confidence of at least 0.5 (default: 0.1)
uv run analyzer analyze "track.m4a" --sonify check.wav      # also write the track with a click at each kick (louder = more confident)
uv run analyzer analyze "track.m4a" -m other.pt             # use a different kick model (default: models/kick.pt)
uv run analyzer analyze "track.m4a" -v                      # show the pipeline steps (-vv: also one line per candidate kick)
uv run analyzer analyze "track.m4a" -q                      # only show warnings and errors
```

The first run downloads the beat_this model (about 78 MB) to the PyTorch cache. Run `uv run analyzer analyze --help` for all options.

Logs go to standard error ([loguru](https://github.com/Delgan/loguru)), so standard output only has the result. The package is silent when you import it as a library; call `logger.enable("analyzer")` to see its logs.

## Training

The kick model learns from synthetic drops made from your own audio samples (not in this repository): kicks, loops and other sounds. A TOML bank config tells which files are kicks, loops, claps, impacts and other hits. Copy `bank.example.toml` to `bank.toml` (git-ignored), set `root` to your samples folder and edit the globs. Then:

```sh
uv run analyzer train bank.toml -v   # about 20 min on an Apple M4 Max (MPS)
```

The first run decodes the samples to `models/bank/` (about 0.7 GB for 3000 samples); a changed config decodes them again. The model with the best validation F-measure goes to `models/kick.pt`. Options: `--cache` (the decoded samples folder), `--steps` (default 15000), `--batch-size` (16), `--workers` (10 processes make the drops), `--device` (`mps`, `cuda` or `cpu`) and `--seed`. `models/` is git-ignored: never commit samples, the bank or models.

Good samples: complete kick one-shots with one kick each (no kick rolls), 160 BPM loops that start on a bar and have no kicks and no sub-bass (screeches, atmospheres, top loops, fills; no drum loops, full mixes or bass stems), and one-shots without a kick layer.

To check the training data, write some synthetic drops and listen to them:

```sh
uv run analyzer synth bank.toml -n 8 -v   # to models/drops/ (git-ignored)
```

Each drop gives `drop_000.wav`, `drop_000.csv` (the exact kick onsets, one time in seconds per line, for example for Sonic Visualiser), `drop_000.clicks.wav` (a click at each kick onset) and `drop_000.json` (the onsets, the sample files and the mix settings). With `-v`, the log also shows the sample files of each drop. Options: `-o` (the output folder), `--seed`, `--held-out` (only held-out kick designs; with the training seed, these are the validation drops) and `--cache`.

## Development

```sh
uv run ruff format   # format
uv run ruff check    # lint
uv run ty check      # type check
```

## Method

1. Decode with ffmpeg to 44.1 kHz, mid channel (L+R)/2.
2. Beat grid: track the beats with [beat_this](https://github.com/CPJKU/beat_this) (`final0` model, CPU, no DBN). Remove extra beats closer than 3/4 of the beat period, fill skipped beats, and extend the beats over the full track. The beat period is the median beat interval, moved by octaves toward 150-170 BPM.
3. Loudness-normalize to -14 LUFS. The kick model gives a kick onset probability for each 10 ms frame. Candidates are the local maxima within +/-20 ms with a probability of at least the minimum confidence.
4. Grid positions, with search windows of +/-40 ms (beat), +/-30 ms (1/8) and +/-15 ms (1/16 at 1/4 and 3/4 of a beat, triplet at 1/3 and 2/3). First the beats are checked. Each beat then moves to its kick if that kick has a probability of at least 0.5 (beat_this gives beats in 20 ms steps), and the other beats move by the median shift. Then the off-beat positions are checked between the moved beats. At each position, the most probable candidate is kept. In each beat, the straight (1/8, 1/16) or the triplet kicks are kept, whichever has the more probable best kick.
5. Remove kicks closer than 40 ms to a more probable kick. The confidence is the model probability. The output is the detected onset time, not the grid time.
6. Beat output: the moved beats from step 4 that are inside the track. A beat with a confident kick is at the kick onset; the other beats are the beat_this beats moved by the median shift.

### Kick model

`model.py`: log-magnitude spectrograms at three window sizes (1024, 2048 and 4096 samples: 23, 46 and 93 ms), 80 mel bands from 27.5 Hz to 16 kHz, 10 ms hop, standardized per band with training statistics. Three 3x3 convolution layers (16, 32 and 32 channels, frequency max-pooling) read the spectrum. Six residual dilated 1D convolution layers (64 channels, dilations 1, 2, 4, 8, 16, 8) add +/-420 ms of context, about one beat on each side. About 110k parameters, one sigmoid output per frame.

`synth.py`: each training example is a synthetic 12 s drop (8 bars at 160 BPM):

- One kick design (all pitched versions of one kick), with a new key every two bars. Each kick cuts the tail of the one before it.
- Kick patterns: beats, 1/8 off-beats, missing beats, single 1/16 and triplet kicks, and rolls of 1, 2 or 4 beats (1/16, triplets or 1/8) at bar ends. Some bars and drops have no kicks.
- Up to three 160 BPM loops (screeches, melody stems, atmospheres, top and ride loops, fills), usually high-passed at 100-250 Hz and ducked by a sidechain curve at each kick. Claps on beats 2 and 4, other one-shots (snares, hats, percussion, FX, synth hits) on the 1/16 grid or off it, and impacts.
- Mastering: EQ tilt, drive into a tanh soft clipper, a peak limiter, then a tempo change of up to +/-6% (150-170 BPM) and a random gain of +/-6 dB after loudness normalization.

The targets are the exact kick onsets (the frame of the onset is 1, its two neighbors 0.5), with binary cross-entropy loss. 10% of the kick designs are held out (split by design, not by file, because the pitched versions are near-duplicates); 200 drops made from them are the validation set.

`bank.py` reads the bank config, lists its samples and decodes them once to a memory-mapped cache.

Known limits: the output is only as good as the beat grid; a wrong beat phase or tempo gives wrong kicks, and kicks off the grid are lost. The model learns only from synthetic drops; real mixes (reverb, layered kicks, other genres) can differ. The confidence ranks kicks but is not a calibrated probability.

Audio files must never be committed (see `.gitignore`).
