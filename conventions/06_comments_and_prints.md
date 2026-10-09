# Comments and prints

Applies to comments, docstrings, `print` and log messages, and markdown cells in notebooks.

## The rules

1. **Lowercase.** Capitals only where a term or symbol needs them: `AUC`, `LITNET-2020`, `Colab`, `FOR`, `Clopper-Pearson`, `N`, file names and code identifiers as written.
2. **Commas and periods only.** No colons, semicolons, dashes, parentheses, quotes, exclamation or question marks, and no emoji in the prose. Symbols that belong to code, paths, numbers, units and math stay as they are: `{path}`, `0.05`, `1e-3`, `x <= t`, `%`.
3. **Natural prose.** Write the way a colleague would explain the line to another colleague. Say what the code does or why it does it that way, not that it exists.
4. **No instructions to a person.** No "remember to", "you should", "note to self", "todo, run this first". Steps for a person belong in `conventions/` or `docs/`. Open questions belong in an issue or in `docs/plan.md`.
5. **Comment the why, not the what.** A comment that repeats the code gets deleted. A comment earns its place by stating an assumption, a source, or a reason that the code cannot show.

## Examples

| Write | Not |
|---|---|
| `# fit on windows 1 to 3, calibrate on window 4` | `# Fit the model: windows 1-3 (train); window 4 (calibrate)!` |
| `# clopper pearson is exact, so the bound holds at any sample size` | `# This is important — it's crucial that we use an exact bound.` |
| `print(f"saved figure {name} to {folder}")` | `print("✅ Done! Figure saved :)")` |
| `print(f"window {w}, prevalence {pi:.4f}, FOR {f:.5f}")` | `print("Window:", w, "| Prevalence =", pi)` |
| `# label shift only, the score distribution within each class is assumed fixed` | `# NOTE: you need to check this assumption later` |

## Prints and logs

- Inside experiments, use `run.log(...)` so the message also lands in `log.txt`.
- One line per event that someone reading the log later would want to find: data loaded with its size, a split, a threshold chosen, a file saved, a run ending. No progress chatter.
- Numbers carry their name: `prevalence 0.0182`, not a bare `0.0182`.
