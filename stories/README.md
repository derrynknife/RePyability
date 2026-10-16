# User stories

The questions people (and the agents working for them) bring to RePyability
that it should be able to answer. These files record *what* is wanted and
*how we will know it is answered*. Units of work go in issues, the larger
ideas and their designs in [ROADMAP.md](../ROADMAP.md), and the outcomes
they serve here.

| File | Area | Stories |
|---|---|---|
| [maintenance-economics.md](maintenance-economics.md) | The net present value of maintenance plans, designs and crews, kept current as the plant changes | MX-01 to MX-25 |

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
  the test that shows it).

## Adding a story

Start from a question someone has actually asked, or a study in the docs
or an issue. Keep the setting concrete and small enough to check by hand
or by an independent calculation. Number new stories after the last one in
the file; do not reuse an ID.
