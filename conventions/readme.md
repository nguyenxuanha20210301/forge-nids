# Conventions

These files are the working rules for forge-nids. When a rule here conflicts with a habit, a template or a default of some tool, the rule here wins. When something is not covered, ask before building, then add the answer to the right file.

| File | Covers |
|---|---|
| [01_workflow.md](01_workflow.md) | The three places (local, GitHub, Drive with Colab), what lives where, the push, pull and run loop |
| [02_git.md](02_git.md) | Who commits, what gets committed, commit messages, what never enters the repository |
| [03_repository_layout.md](03_repository_layout.md) | Where each kind of file goes in the repository |
| [04_drive_layout.md](04_drive_layout.md) | Folders on Drive, run folders, naming and immutability |
| [05_code.md](05_code.md) | File names, paths, configs, seeds, dependencies |
| [06_comments_and_prints.md](06_comments_and_prints.md) | How comments, docstrings, prints and notebook text are written |
| [07_results_and_figures.md](07_results_and_figures.md) | What a run saves, how figures are drawn and exported |
| [08_notebooks.md](08_notebooks.md) | Notebooks as thin runners, uploading to Drive, clearing outputs |
| [09_writing.md](09_writing.md) | Prose in docs and the paper |

## Hard rules

1. Everything is in English: code, comments, prints, figures, docs, commit messages.
2. Data and run outputs live on Drive. The repository holds code, configs, notebooks, conventions and small paper results. The proposal, the plan and other notes in `docs/` stay on the local machine and are never committed.
3. The author runs every git command. Nothing else commits or pushes, and no commit carries a co-author or generated-by line.
4. No machine-specific path, username, email or token appears in any committed file.
5. All file and folder names are lowercase.
6. Comments and prints are lowercase except terms and symbols, written as natural prose, with commas and periods as the only punctuation.
7. Every figure can be redrawn by a script from saved results, without rerunning the experiment.
8. Figures carry English text only and no title or description. Each one is saved as a 600 dpi png, a pdf, and a pdf filling one A4 page.
