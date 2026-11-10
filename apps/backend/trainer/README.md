# Track analysis

Run from this directory:

```sh
uv run trainer analyze songs/2004_MN4.mp3 -m model-rolls.pt \
  --labels songs/2004_MN4-labels.csv -o results/2004_MN4-onsets.csv
```

Create the output directory before you run the command. Without `-o`,
`analyze` writes `<audio-name>-onsets.csv` beside the audio, leaving
`<audio-name>.csv` label files unchanged. An explicit `-o` path is used
exactly as supplied. With `--clicks`, the default click file is
`<audio-name>-onsets.clicks.wav`.

With `--labels`, scoring splits the labels at gaps longer than 5 seconds.
Each scored span starts at its first label minus 20 ms and ends at its last
label plus 20 ms. Predictions outside these spans do not count as false
positives. Pauses of 5 seconds or less remain inside the scored span.
The onset match tolerance is 20 ms. An empty label file has no scored spans.

The output CSV still contains predictions for the full track. Span selection
affects scoring only. Each model uses its saved threshold unless you supply
`--threshold`.

These spans are inferred from onset times, not exact drop boundaries. A pause
longer than 5 seconds inside a labeled drop is excluded by this rule.

# Real-data fine-tuning

`src.services.training.finetune.fine_tune` starts from an existing checkpoint
and uses equal numbers of real and synthetic examples in each batch. It keeps
the checkpoint's feature normalization and uses the existing kick-attack trim
for synthetic examples. Frames outside labeled spans and crop-edge context
are excluded from the loss.

Supply separate lists of training and validation audio paths. Each audio
file must have exactly one matching `<name>.csv` or `<name>-labels.csv`.
Validation selects the checkpoint and peak threshold by combined F-score
inside labeled spans. The starting checkpoint is also a selection candidate,
so a run with no validation improvement retains the starting weights.
The output path must be new.

Keep test tracks separate from both lists. Select the baseline's threshold
on the same validation tracks, then compare both checkpoints on the test
tracks with their selected thresholds. Record the split, seed, training and
synthesis settings with each run. A single run does not measure seed variation.