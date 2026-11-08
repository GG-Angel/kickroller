# Track analysis

Run from this directory:

```sh
uv run trainer analyze songs/2004_MN4.mp3 -m model-rolls.pt \
  --labels songs/2004_MN4-labels.csv -o results/2004_MN4.csv
```

Create the output directory before you run the command. Always use `-o` when
the track's CSV contains hand labels, to keep those labels unchanged.

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