# Writing

Applies to `docs/`, `readme.md` files, commit messages and the paper. Comments and prints follow [06_comments_and_prints.md](06_comments_and_prints.md) on top of this.

Prose here should read as written by someone who did the work and can be wrong about it. Fluent but generic text is the default output of autopilot, whoever produced it. Specific text takes a person who knows the details.

## What to put in

- **Specifics.** Exact numbers from the runs (0.0183, not "about 2%"), the dataset and window, the run id, the library version when it matters, the error message as it appeared.
- **A date and a state.** Notes in `docs/` say when they were written and what is still open: "as of 2026-10-14, UGR16 weeks 18 to 23 are not downloaded yet".
- **Positions.** Say which option was picked and why, and what would change the decision. Leave a failure as a failure when it was one.
- **Uneven depth.** Spend words on what is hard or surprising and a line on what is routine. Sections do not need equal length.
- **A known reader.** Write for someone who has read the proposal. Do not re-explain conformal prediction in every note.

## What to leave out

- Signposting: "In this section we will", "Let us now turn to", "In summary".
- Closing paragraphs that restate the body.
- Hedges stapled to every claim: "it is worth noting", "arguably", "it is important to". Hedge the claims that are uncertain, and say why.
- Stock words that carry nothing: delve, crucial, pivotal, robust (unless it means a robustness guarantee), seamless, landscape, leverage, novel (unless it is defended against related work), comprehensive.
- Negation-contrast frames ("this is not X, it is Y"), lists of exactly three for rhythm, a rhetorical question answered in the next sentence.
- Vague attribution: "studies show", "it is well known". Cite the paper.
- Em dashes. Use a comma, a period or parentheses.

## Paper specifics

- Every claim of novelty names the closest prior work and the difference, as in the "sentences not to write" list in `docs/plan.md`.
- Limitations are concrete: which assumption, which dataset, which window, how large the effect.
- Numbers in the text match a file in `results/` exactly.

## Checking a draft

Before a note or a section is considered done, ask three questions of it.

1. Could this paragraph sit in another paper on the same topic without edits? If so, it says nothing about this work.
2. Is there a sentence that could turn out wrong? If none, the text is not committing to anything.
3. Take the three most specific details. Are they used again later? Details that appear once and vanish were decoration.
