# OWASP Mapper Eval Harness

Internal developer tooling for v2.0. **Not a user-facing feature.**

This harness measures whether changes to the OWASP-Agentic-AI mapper
(either the deterministic rules in `agent_recon.pt.owasp_mapper` or
the LLM-driven Mapper agent prompt) make the mapper better or worse
against a hand-labeled ground-truth set. The goal is numbers, not
vibes.

---

## TL;DR

```bash
# Default: evaluate the deterministic rule-based mapper.
ai-agent-recon eval-mapper \
    --fixtures-dir evals/ground_truth \
    --output evals/results

# Optional: evaluate the CrewAI Mapper agent end-to-end.
ai-agent-recon eval-mapper --llm \
    --fixtures-dir evals/ground_truth \
    --output evals/results

# As a CI gate (fail the build if macro F1 drops below 0.9):
ai-agent-recon eval-mapper --fail-under 0.9
```

Three artifacts land in `--output`:

| File | Purpose |
|------|---------|
| `eval_results.json` | Full per-fixture + aggregate metrics. Canonical artifact; everything else is derived. |
| `eval_results.csv`  | One row per `(fixture, ASI)` cell. Designed to `diff` cleanly between runs. |
| `eval_results.md`   | Operator-readable summary with per-fixture pass/fail and the per-ASI table. |

---

## How fixtures work

A fixture is a pair of files in `evals/ground_truth/`:

  * `<name>.json`           — a recon input (NormalizedRecon OR a
                              Phase-1 FinalReport), exactly what
                              `ai-agent-recon pt-plan -i` would accept.
  * `<name>.expected.json`  — hand-labeled ground truth for the input.

The expected file must cover **every** ASI category (ASI01..ASI10),
exactly once. Each entry looks like this:

```json
{
  "asi_id": "ASI06",
  "applicable": true,
  "expected_priority": "Critical",
  "expected_signals_must_contain": ["long-term"],
  "note": "RAG + long-term memory — canonical context-poisoning target."
}
```

Fields:

  * `applicable`                       — required. Must the mapper
                                         mark this category as in-scope?
  * `expected_priority`                — required when `applicable=true`,
                                         else null. Ladder: Informational
                                         < Low < Medium < High < Critical.
  * `expected_signals_must_contain`    — optional list of substrings.
                                         At least one of the mapper's
                                         emitted signals for this ASI
                                         must contain each substring
                                         (case-insensitive). Useful for
                                         locking down specific evidence
                                         (e.g. "has_mcp" for an MCP
                                         target).
  * `note`                             — free text, for humans only.

A top-level `priority_tolerance` controls how strictly priority is
compared:

  * `exact`     — must match exactly.
  * `one_step`  — may differ by one ladder step. **Default.** Captures
                  "right ballpark" without punishing weight-tuning.
  * `any`       — priority is not scored at all; only applicability.

---

## What the metrics mean

We score two things independently:

### 1. Applicability — primary signal

Per-ASI binary classification, aggregated into:

  * **Per-ASI** precision / recall / F1 — diagnose which categories
    the mapper consistently gets right or wrong.
  * **Micro** P/R/F1 — single number across all (fixture × ASI) cells.
  * **Macro** P/R/F1 — unweighted mean across the ten ASIs, so a rare
    category isn't drowned out by the common ones.

Applicability is primary because if it's wrong, the operator either
misses a category or wastes time on irrelevant ones.

### 2. Priority agreement — secondary signal

For cells where both predicted and expected are applicable:

  * Exact-match rate
  * Within-one-step rate (under the default tolerance)
  * Mean Manhattan distance on the priority ladder

Priority is secondary because it's the noisier signal. Use it to spot
systematic bias (e.g. "the mapper inflates everything to Critical")
rather than as a hard CI gate.

### 3. Per-fixture pass/fail

A fixture passes iff *every* comparison is applicability-correct AND
every applicable category passes its priority-tolerance check AND
every signal substring check holds. The CLI prints `N/M fixtures
passed` and a list of failures with human-readable reasons.

---

## When to add a fixture

Add a new fixture when:

  * You spot a real-world recon shape that the current ground-truth
    set doesn't cover.
  * You investigate an operator's bug report and conclude the mapper
    was indeed wrong — capture the reproduction here.
  * You change the mapper rules in a way you want to lock down.

To add one:

  1. Drop the recon JSON at `evals/ground_truth/<name>.json`. It's the
     same shape `ai-agent-recon pt-plan -i` accepts.
  2. Write `<name>.expected.json` by hand. Use the existing fixtures
     as templates.
  3. Run `ai-agent-recon eval-mapper` and confirm the new fixture
     parses (the loader fails loud if it doesn't).
  4. Commit both files together.

---

## Mode: rule-based vs LLM

  * **Rule-based (default)** — calls `map_owasp()` directly. No LLM,
    no API key, fully reproducible. Use this for the CI gate.
  * **`--llm`** — drives the CrewAI Mapper agent through the PT crew
    and pulls just the mapping out. Per-fixture failures fall back to
    the rule-based mapper so a single provider hiccup doesn't wipe out
    the whole run.

The two modes share the same fixtures, schema, and metrics — so a
direct number-to-number comparison is meaningful. If LLM mode beats
rule-based by 3 macro-F1 points on the same fixtures, that's
real signal that the prompt is doing useful work.

---

## Current baseline (v2.0)

Running `eval-mapper` against the six checked-in fixtures with the
deterministic mapper shows:

  * Applicability: macro F1 ≈ 0.95 (mapper is good at deciding what's
    in-scope).
  * Priority within-one-step: ≈ 63% (mapper is too aggressive — most
    applicable categories get bumped to Critical regardless of risk).

The priority-inflation finding is exactly the kind of signal this
eval is built to surface: it didn't require eyeballing reports, it
fell straight out of the numbers. Closing that gap is one of the
follow-on PRs we'd plan after v2.0 ships.
