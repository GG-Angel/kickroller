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
uv run analyzer analyze "track.m4a" -c 0.2                  # also kicks with a confidence from 0.2 to 0.5 (default: 0.5)
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

The first run decodes the samples to `models/bank/` (about 0.7 GB for 3000 samples); a changed config decodes them again. The model with the best validation F-measure goes to `models/kick.pt`. Options: `--cache` (the decoded samples folder), `--steps` (default 15000), `--batch-size` (16), `--workers` (10 processes make the drops), `--device` (`mps`, `cuda` or `cpu`), `--seed` and `--all-designs` (also train on the held-out kick designs, for a final model after tuning; the validation F is then too high). The option defaults and the other training values come from the [settings](#settings). `models/` is git-ignored: never commit samples, the bank or models.

Good samples: complete kick one-shots with one kick each (no kick rolls), loops that start on a bar and have no kicks and no sub-bass (screeches, atmospheres, top loops, fills; no drum loops, full mixes or bass stems), and one-shots without a kick layer. Samples must be 160 BPM, or you give their tempo with `bpm` (any section, for example `[kick.bpm]`); then they are time-stretched to 160 BPM with the same pitch. This uses ffmpeg `atempo` (WSOLA), which keeps kick attacks sharp: on 155 BPM kicks the first 15 ms stay the same, and the model still finds each kick. Rubber Band smeared the attacks.

To check the training data, write some synthetic drops and listen to them:

```sh
uv run analyzer synth bank.toml -n 8 -v   # to models/drops/ (git-ignored)
```

Each drop gives `drop_000.wav`, `drop_000.csv` (the exact kick onsets, one time in seconds per line, for example for Sonic Visualiser), `drop_000.clicks.wav` (a click at each kick onset) and `drop_000.json` (the onsets, the sample files and the mix settings). With `-v`, the log also shows the sample files of each drop. Options: `-o` (the output folder), `--seed` (by default a new seed each run; the log shows it, so you can make the same drops again), `--held-out` (only held-out kick designs; with `--seed` set to the training seed, 0 by default, these are the validation drops) and `--cache`.

## Settings

All values that you can tune have a default in `src/analyzer/settings.py`, in four sections: `detector` (peak picking and the beat grid), `model` (the kick CNN for new models; a saved model keeps its own model settings), `training`, and `synth` (the random ranges and chances of the synthetic drops). To change a value without a code change, set an environment variable or write it in a `.env` file in this folder (git-ignored). The name is `ANALYZER_`, then the section and the field with `__` between them. Lists and tables are JSON:

```sh
ANALYZER_DETECTOR__MIN_CONFIDENCE=0.4
ANALYZER_TRAINING__VALIDATION_DROPS=400
ANALYZER_SYNTH__KICKS__ROLL_CHANCE=0.25
ANALYZER_SYNTH__LOOPS__LEVEL_DB='[-20, 0]'                    # [low, high]
ANALYZER_SYNTH__KICKS__ROLL_BEATS='{"1": 0.6, "2": 0.4}'      # {value: chance}; the chances add up to 1
ANALYZER_SYNTH__TEMPOS='[[0.6, 160, 160], [0.4, 150, 170]]'   # [chance, low BPM, high BPM]
```

A command-line option (for example `-c` or `--steps`) changes its value for one run, and its default comes from these settings. A wrong value stops the command with an error: a wrong field name in a section, a range with the high value first, or chances that do not add up to 1. A wrong section name has no effect. Changes to the `synth` settings also change the drops that a seed gives.

## Development

```sh
just check   # ty, Ruff lint and formatting checks
just fix     # apply Ruff fixes, then run all checks
```

The code in `src/analyzer/` is in five packages. Each package only uses the packages before it in this list:

1. `audio/`: decoding with ffmpeg, the mid channel, loudness and WAV files (`io.py`), and click tracks (`sonify.py`).
2. `model/`: the kick CNN. The input features (`features.py`), the network (`network.py`), and how to save, load and run a model (`checkpoint.py`).
3. `detection/`: from a track to kicks and beats. The pipeline (`detector.py`), the beat grid (`grid.py`) and peak picking (`peaks.py`).
4. `training/`: the sample bank (`bank.py`), the synthetic drops (`synth.py`) and the training loop (`train.py`).
5. `cli/`: one module for each command (`analyze.py`, `train.py`, `synth.py`), and the shared options and logging (`options.py`).

All packages can use `settings.py` (the [settings](#settings)). Values that are not settings are named constants at the top of their module. Functions whose names start with `_` are only for use in their own module.

As a library, `analyzer` exports `detect_kicks`, `detect_kicks_in_signal`, `Detection`, `DetectorSettings` and `Settings`.

## Method

1. Decode with ffmpeg to 44.1 kHz, mid channel (L+R)/2.
2. Beat grid: track the beats with [beat_this](https://github.com/CPJKU/beat_this) (`final0` model, CPU, no DBN). Remove extra beats closer than 3/4 of the beat period, fill skipped beats, and extend the beats over the full track. The beat period is the median beat interval, moved by octaves toward 150-170 BPM (so tempos from about 113 to 226 BPM stay as they are).
3. Kicks: loudness-normalize to -14 LUFS. The kick model gives a kick onset probability for each 10 ms frame. The kicks are the local maxima within +/-20 ms with a probability of at least the minimum confidence (default 0.5: on synthetic drops, F 0.985 against 0.973 at 0.1). A kick closer than 1/10 of a beat (80% of a 1/32 note at the beat grid tempo), but at most 40 ms, to a more probable kick is removed. The time of each kick is refined between frames by quadratic interpolation: a parabola through the log probability of the peak frame and its two neighbors (on synthetic drops, this reduces the timing error from 3.1 to 2.6 ms std). The confidence is the model probability.
4. Beat output: beat_this gives beats in 20 ms steps, so each beat moves to its most probable kick within +/-40 ms if that kick has a probability of at least 0.5. The other beats move by the median shift. The output is the beats inside the track.

The kicks only use the tempo of the beat grid, for the minimum distance in step 3. The 40 ms limit makes this safe: a wrong tempo can only make the rule shorter, and the +/-20 ms local maximum still allows only one kick per peak. At 150-170 BPM the rule is 40 ms (4 frames); at 200 BPM it is 30 ms, so 1/32 rolls (37.5 ms) are kept. The kicks do not use the beat positions. On synthetic drops, a filter that kept only the kicks near the beats, 1/8, 1/16 and triplet positions of the grid removed 14 false kicks and 888 real kicks (recall 0.98 to 0.86), because beat_this is sometimes off by a few 10 ms or has a wrong tempo in a section.

### Kick model

`model/`: log-magnitude spectrograms at three window sizes (1024, 2048 and 4096 samples: 23, 46 and 93 ms), 80 mel bands from 27.5 Hz to 16 kHz, 10 ms hop, standardized per band with training statistics. Three 3x3 convolution layers (16, 32 and 32 channels, frequency max-pooling) read the spectrum. Six residual dilated 1D convolution layers (64 channels, dilations 1, 2, 4, 8, 16, 8) add +/-420 ms of context, about one beat on each side. About 110k parameters, one sigmoid output per frame.

`training/synth.py`: each training example is a synthetic 12 s drop (8 bars at 160 BPM) at a whole-number tempo: 160 BPM in 50% of the drops, 150-159 BPM in 25%, 161-170 BPM in 15% and 171-200 BPM in 10%. All parts are put on the grid of this tempo:

- One kick design (all pitched versions of one kick), with a new key every two bars. Each kick cuts the tail of the one before it. The kicks and one-shots keep their sound at each tempo.
- Kick patterns: beats, 1/8 off-beats, missing beats, single 1/16 and triplet kicks, and rolls of 1, 2 or 4 beats (1/16, triplets or 1/8) at bar ends. Some bars and drops have no kicks.
- Up to three loops (screeches, melody stems, atmospheres, top and ride loops, fills), resampled from 160 BPM to the drop tempo (this also changes their pitch, from -1.1 semitones at 150 BPM to +3.9 at 200 BPM), usually high-passed at 100-250 Hz and ducked by a sidechain curve at each kick. Claps on beats 2 and 4, other one-shots (snares, hats, percussion, FX, synth hits) on the 1/16 grid or off it, and impacts.
- Mastering: EQ tilt, drive into a tanh soft clipper, a peak limiter, and a random gain of +/-6 dB after loudness normalization.

The targets are the exact kick onsets (the frame of the onset is 1, its two neighbors 0.5), with binary cross-entropy loss. 10% of the kick designs are held out (split by design, not by file, because the pitched versions are near-duplicates); 200 drops made from them are the validation set. With `--all-designs`, training also uses the held-out designs (for a final model after tuning); the validation F is then too high.

`training/bank.py` reads the bank config, lists its samples and decodes them once to a memory-mapped cache.

Known limits: the beat output is only as good as beat_this; a wrong beat phase or tempo gives wrong beats (the kicks change at most through the minimum distance). The model learns only from synthetic drops at 150-200 BPM (mostly 160 BPM); real mixes (reverb, layered kicks, other genres) can differ. A model trained only on 150-170 BPM drops found 87 of the 93 kicks of a track sped up from 160 to 170 BPM, but only 63 at 200 BPM; this is why the drops now go up to 200 BPM. The confidence ranks kicks but is not a calibrated probability.

Audio files must never be committed (see `.gitignore`).
