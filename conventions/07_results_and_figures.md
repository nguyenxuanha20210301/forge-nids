# Results and figures

## Save everything a figure needs

A run saves every number any of its figures or tables uses, in `results/`, before drawing anything. The test is simple: delete `figures/`, run `python scripts/plot.py <run_dir>`, and every figure comes back identical.

| Kind | Format | Saved with |
|---|---|---|
| tables, per-window metrics, curves | `parquet`, plus a `csv` copy for quick reading | `run.save_table(df, name)` |
| arrays, scores, labels, predictions | compressed `npz` | `run.save_arrays(name, scores=..., labels=...)` |
| scalars, chosen thresholds, summaries | `json` | `run.save_json(obj, name)` |

Per-flow scores and labels are saved whenever they fit, because most later questions are answered from them without retraining. Model checkpoints go in `results/models/` when retraining would take more than a few minutes.

## Draw during the run, from saved files

Figures are drawn on Colab at the end of each run, into `figures/` of the run folder. Each experiment module lists its figures in `FIGURES`, a dict from figure name to a function that takes the run folder and returns a matplotlib figure. The function reads only from `results/`, never from variables in memory. That one rule is what lets `scripts/plot.py` redraw any figure later.

```python
def fig_for_by_window(run_dir):
    style.use()
    curves = runs.load_table(run_dir, "curves")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.4))
    ...
    return fig

FIGURES = {"for_by_window": fig_for_by_window}
```

Figures that combine several runs, such as a comparison across datasets, are drawn locally by a script in `analysis/` that reads the pulled run folders and saves into `results/figures/`.

## Figure rules

- **Text.** English only. Axis labels, tick labels, legends and panel letters are allowed. No title, no caption, no explanatory note inside the figure. The paper caption says what the figure shows. `save_figure` refuses a figure with a title.
- **Formats.** `save_figure(fig, folder, name)` writes three files.
  - `<name>.png` at 600 dpi.
  - `<name>.pdf`, vector, at the size it will have in the paper.
  - `<name>_a4.pdf`, the same vector figure scaled to fill one A4 page with 15 mm margins, landscape for wide figures and portrait for tall ones.
- **Size.** Draw at the final printed size, `style.WIDTH_SINGLE` (3.5 in) or `style.WIDTH_DOUBLE` (7.16 in), so font sizes in the paper are the ones set in `style.py`.
- **Color.** A method keeps the same color in every figure. Colors come from `style.COLORS` by a fixed method order, never by rank. Lines that differ only by color also differ by marker or dash when the figure might be printed in gray.
- **Names.** Lowercase, describing the content, not the figure number: `for_by_window`, `dismiss_vs_for`. Figure numbers change while writing, names do not.

## Results in the repository

`results/` in the repository holds only what the paper uses: final tables as `csv` and final figures in the three formats. Each file there starts from a run folder, and `results/readme.md` lists, for each file, the run id it came from.
