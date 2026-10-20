# Dataset location

Put the training data for your classifier in this folder.

## Format

A UTF-8 CSV with a header row and one example per row:

| column | meaning |
|---|---|
| `text` | the sentence or utterance to classify |
| `label` | the class name, any string (for example `relevant`, `off_topic`) |

```csv
text,label
Fake sample text 01: placeholder sentence for class_a.,class_a
Fake sample text 01: placeholder sentence for class_b.,class_b
```

The column names are configurable (`text_column`, `label_column` in `config/blueprint.toml`).

## Rules the loader checks

- Both columns exist and no `text` or `label` value is blank.
- At least **2 classes**, and at least **3 rows per class** (so each class can appear in train, validation
  and test).
- The labels are collected and sorted automatically; you do not need to number them.
- The split is stratified and reproducible: `val_fraction` and `test_fraction` of each class are held out
  (default 20% each, seed 42). The test split is only used for the final evaluation and the quantization
  comparison, never for choosing the best epoch.

## `sample.csv`

`sample.csv` is a **fake placeholder**: 45 rows, 3 fake classes (`class_a`, `class_b`, `class_c`), no real
content. It only exists so the whole pipeline (train, export, quantize, predict) can be run end to end.
Models trained on it are meaningless. Replace it with your own file and point `[data] path` in the config at it.

## Git

Everything in this folder is git-ignored except `sample.csv` and this README, so real datasets stay on your
machine. Do not commit data that is private or covered by a licence you cannot redistribute.
