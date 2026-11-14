# User stories

The questions people (and the agents working for them) bring to RePyability
that it should be able to answer. These files record *what* is wanted and
*how we will know it is answered*.

| File | Area | Stories |
|---|---|---|
| [maintenance-economics.md](maintenance-economics.md) | The net present value of maintenance plans, designs and crews, kept current as the plant changes | MX-01 to MX-25 |

The maintenance questions of one part or failure mode, asked in model time
(when to replace a part next, a skipped or late replacement, the numbers
behind each maintenance decision), are kept as use cases in
[`repyability/maintenance/README.md`](../repyability/maintenance/README.md),
each written down and answered by hand before it is built.

## Format

Each story has

- an **ID** (`MX-03`), stable once written, so code, tests and issues can
  cite it;
- a **persona**: who asks, and in what job;
- the **story**: *as a ..., I want ..., so that ...*;
- the **question in their words**, as they would type it to RePyability
  or to an agent;
- the **setting**: a concrete system, taken where possible from the docs,
  the tutorial or a study already in the repository, so the story can be
  checked against numbers we know;
- **acceptance criteria**: what an answer must contain, and the
  independent checks (an oracle) it must pass, so the answer is known to
  be right and not just present;
- what it **needs**: the building blocks (listed at the top of each file)
  it depends on;
- **references**, where a standard text sets out the question;
- a **status**: *open* (not answerable without the analyst writing it by
  hand), *partial* (part of it can be done today, said how) or *met* (with
  the test that shows it), and the release it is in (or "on `dev` for"
  the next).

## Oracles and statuses

A story's oracle is an ordinary test in `repyability/tests/`, cited from
the story as `file::test_name`: the independent check its acceptance
criteria name (a closed form, a calculation by hand stated in the test, a
chain or a sum built in the test without the package's code, or a
seeded simulation against the exact value). There is no separate
scenario suite. When a pull request builds part or all of a story, it
moves the status to *partial* or *met*, says what is left, and cites the
tests that check what it built; a story is *met* only when every
acceptance criterion has its test. After each release, the statuses are
read again against what shipped, and each persona's question is asked of
the release as they would ask it.

## Adding a story

Start from a question someone has actually asked, or a study in the docs
or an issue. Keep the setting concrete and small enough to check by hand
or by an independent calculation. Number new stories after the last one in
the file; do not reuse an ID.
