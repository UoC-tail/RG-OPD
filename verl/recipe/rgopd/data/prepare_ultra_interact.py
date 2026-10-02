#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 The RG-OPD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download and preprocess `openbmb/UltraInteract_sft` into the train/val parquet files
RG-OPD trains on.

Appendix A.1 of the paper (Akhondzadeh et al., "Reward-Gated On-Policy Distillation",
arXiv:2607.04037) trains on "a subset of UltraInteract". This script produces that
subset and converts it into this repo's standard parquet schema (`prompt`, `data_source`,
`ability`, `reward_model`, `extra_info`), matching what `recipe/rgopd/config/
rgopd_trainer.yaml`'s `data.train_files` / `data.val_files` expect.

Task filtering (the one substantive behavior change vs. a plain schema conversion)
------------------------------------------------------------------------------------
UltraInteract_sft's `task` column takes one of four values: `Coding`, `Math_CoT`,
`Math_PoT`, `Logic`. This recipe's reward verifiers
(`recipe/rgopd/reward_score/math.py`, `recipe/rgopd/reward_score/code.py`, dispatched by
`recipe/rgopd/reward_score/__init__.py`) only know how to score two `data_source`
values: `"math"` (boxed-answer checking) and `"code"` (sandboxed test-case execution).
There is no verifier for `Logic` -- and the paper's own evaluation suite (GSM8K, MATH,
MBPP, etc., Appendix A.1) is math/code/knowledge, not logic-puzzle-specific -- so by
default this script keeps only `Coding`, `Math_CoT`, and `Math_PoT` rows, mapped to
`data_source="code"` / `data_source="math"` respectively (NOT the single constant
`"ultra_interact_sft"` the original prep script used internally, which doesn't match
anything in the dispatch table above).

Ground truth availability (read before you rely on the `Coding` split)
------------------------------------------------------------------------------------
`UltraInteract_sft` (the SFT/behavior-cloning split of UltraInteract -- as opposed to
e.g. its preference-pair splits) has exactly six columns: `task`, `dataset`,
`instruction`, `response`, `id`, `parent_id`. It does NOT carry a separate
machine-checkable answer field for any task, so ground truth has to be extracted from
free text -- how reliably that works varies a lot by task and, for `Coding`, by which
of the four `dataset` sub-sources (`codecontest`, `TACO`, `magicoder_evol_instruct`,
`wiki_table_questions`) a row comes from. Everything below was checked empirically
against a live sample of the dataset (see `_extract_boxed_answer`/`_extract_stdio_examples`
docstrings for the exact patterns), not assumed from field names:
  * `Math_CoT` responses reliably end in a LaTeX `\\boxed{...}` answer, so this script
    extracts that as `reward_model.ground_truth` for the `math` verifier. Rows without a
    boxed answer are dropped.
  * `Math_PoT` responses are Python snippets ending in a `print(...)` call, not a boxed
    answer -- so in practice almost none of them pass the same boxed-answer extraction,
    and almost all `Math_PoT` rows get dropped. `Math_PoT` is kept in the default
    `--tasks` for parity with the paper's task list, but expect its contribution to the
    final dataset to be small to none.
  * `Coding` / `dataset=="codecontest"` (competitive-programming problems, ~39% of
    Coding, 44,662 rows) reliably embed one or more illustrative "Examples\n\nInput\n\n
    ...\n\n\nOutput\n\n..." (or "SAMPLE INPUT\n...\n\nSAMPLE OUTPUT\n...") blocks in the
    instruction. This script parses those into `code.py`'s `{"testtype": "stdin", ...}`
    ground-truth schema. Validated by checking whether the dataset's OWN expert
    `response` actually passes the examples extracted from its own instruction (it
    should, if parsing is correct): on a random 300-row sample, 93% parsed successfully,
    and of those, the expert response scored a full pass 72.5% of the time, with another
    ~26% "ran correctly but printed a different (also plausibly valid) answer" -- the
    well-known "problem accepts multiple correct outputs, we only have one" limitation of
    exact-match judging in competitive programming, not evidence of broken parsing -- and
    only ~2% genuine harness/format errors. Pass `--verify-code-examples` to have this
    script re-run that same check on every candidate row at prep time and drop any row
    whose own expert response doesn't pass its extracted examples, trading a slower run
    for a cleaner (but strictly smaller) kept set; see `--verify-code-examples` below.
  * `Coding` / `dataset in {"TACO", "magicoder_evol_instruct", "wiki_table_questions"}`
    are dropped. `TACO`'s ~58,188 rows looked promising at a glance (some rows show the
    same "Sample Input/Output" pattern as codecontest) but turned out NOT to be safely
    parseable: excluding the ~11% that are explicitly function-signature tasks (need a
    `fn_name` this script doesn't extract), only ~6% of the remainder even match the
    stdin-block pattern, and of THOSE, the expert response passed its own extracted
    examples only ~5% of the time (mostly `SyntaxError`/`EOFError` -- the "response" is
    often not a complete, directly-runnable stdin program for this sub-source). Scraping
    it would risk exactly the silently-wrong rewards this script is trying to avoid.
    `magicoder_evol_instruct` (free-form instructions) and `wiki_table_questions`
    (SQL-over-a-table questions, a different verification modality entirely) don't carry
    example I/O at all in the rows sampled. If you need more code data than
    `codecontest` provides, curate/merge a dataset that carries real test cases (e.g. one
    of the APPS/LiveCodeBench releases, or TACO's *original* release --
    `likaixin/TACO` -- which does carry full structured test cases that UltraInteract's
    conversion to instruction/response format dropped) and write its own
    `data_source="code"` rows in this schema.

`Logic` has no verifier in this recipe at all (see `recipe/rgopd/reward_score/`), so
`--tasks` refuses it outright rather than silently letting reward computation fail
partway through a training run -- see `--tasks` below.

Usage
------------------------------------------------------------------------------------
    python -m recipe.rgopd.data.prepare_ultra_interact \\
        --output-dir ./data/rgopd \\
        --val-ratio 0.01 \\
        --seed 42

    # Quick smoke test on a small slice of the dataset:
    python -m recipe.rgopd.data.prepare_ultra_interact \\
        --output-dir ./data/rgopd-smoke \\
        --max-samples 200

Run from the `verl/` directory (this repo's vendored framework root), matching the rest
of this recipe's launch convention (see `recipe/rgopd/README.md`).
"""

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# Default per-test-case timeout (seconds) for extracted codecontest ground truth; passed
# through to reward_score/code.py's compute_score as `test_cases["time_limit"]`. A bit
# more generous than code.py's own DEFAULT_TIMEOUT=1s fallback, since some competitive-
# programming inputs are non-trivial to process.
CODECONTEST_TIME_LIMIT_S = 4.0

# `task` -> `data_source` actually understood by
# `recipe/rgopd/reward_score/__init__.py`'s dispatch table.
TASK_TO_DATA_SOURCE = {
    "Coding": "code",
    "Math_CoT": "math",
    "Math_PoT": "math",
}

# `task` -> descriptive `ability` label (metadata only; not used for verifier dispatch,
# which goes by `data_source`).
TASK_TO_ABILITY = {
    "Coding": "coding",
    "Math_CoT": "math",
    "Math_PoT": "math",
}

# Tasks present in UltraInteract_sft with no verifier in this recipe at all.
UNSUPPORTED_TASKS = {"Logic"}

ALL_KNOWN_TASKS = set(TASK_TO_DATA_SOURCE) | UNSUPPORTED_TASKS

DEFAULT_TASKS = "Coding,Math_CoT,Math_PoT"


def _parse_tasks_arg(raw: str) -> list[str]:
    """Parse and validate `--tasks`, refusing `Logic` (and any unknown value) outright."""
    tasks = [t.strip() for t in raw.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("--tasks: no tasks given")

    unknown = [t for t in tasks if t not in ALL_KNOWN_TASKS]
    if unknown:
        raise SystemExit(
            f"--tasks: unrecognized task(s) {unknown}. UltraInteract_sft's `task` "
            f"column only takes values from {sorted(ALL_KNOWN_TASKS)}."
        )

    requested_unsupported = [t for t in tasks if t in UNSUPPORTED_TASKS]
    if requested_unsupported:
        raise SystemExit(
            f"--tasks: {requested_unsupported} requested, but this recipe ships no "
            "verifier for it. recipe/rgopd/reward_score/ only contains a math verifier "
            "(math.py, boxed-answer checking) and a code verifier (code.py, sandboxed "
            "test-case execution) -- see recipe/rgopd/reward_score/__init__.py's "
            "dispatch table. Forcing Logic rows through this script would produce rows "
            "whose data_source has no matching verifier, and reward computation would "
            "raise at training time the first time one is sampled. Add a Logic verifier "
            "to recipe/rgopd/reward_score/ (and its dispatch table) first if you want "
            "to train on it."
        )

    return tasks


def _extract_boxed_answer(response: str) -> Optional[str]:
    """Extract the content of the last `\\boxed{...}` in `response`, or None if absent.

    Mirrors the brace-matched extraction in `recipe/rgopd/reward_score/math.py`
    (`last_boxed_only_string` + `remove_boxed`), duplicated here in miniature rather
    than imported so this data-prep script doesn't pull in that module's `math_verify`
    dependency just to find a substring.
    """
    idx = response.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    depth = 0
    while i < len(response):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    left = r"\boxed{"
    return response[idx + len(left) : right_brace_idx]


def _extract_stdio_examples(instruction: str) -> Optional[tuple[list[str], list[str]]]:
    """Extract stdin/stdout example test cases from a `codecontest`-style instruction.

    Handles the two labeling conventions actually observed in `UltraInteract_sft`'s
    `codecontest` rows:
      - "Examples\\n\\nInput\\n\\n<in>\\n\\n\\nOutput\\n\\n<out>" (repeated for multiple
        examples, optionally followed by a "Note\\n\\n<explanation>" section to ignore)
      - "SAMPLE INPUT\\n<in>\\n\\nSAMPLE OUTPUT\\n<out>" (older-style problems)

    Returns `(inputs, outputs)`, each a list of one string per example with a trailing
    newline (matching what `reward_score/code.py`'s stdin test runner expects), or `None`
    if neither pattern is found or the two lists would end up mismatched in length.

    Deliberately narrow: this is only applied to `dataset=="codecontest"` rows (see
    `convert_row` and the module docstring's "Ground truth availability" section for why
    the same approach is NOT safe to use on the other `Coding` sub-sources).
    """
    m = re.search(r"\n(?:Examples?|SAMPLE INPUT)\n", instruction, re.IGNORECASE)
    if m is None:
        return None
    tail = instruction[m.start() :]
    tail = re.split(r"\nNote\n", tail)[0]  # drop any trailing explanation section

    pairs = re.findall(r"(?:^|\n)Input\n\n(.*?)\n\n\nOutput\n\n(.*?)(?=\n\n\nInput\n\n|\Z)", tail, re.DOTALL)
    if not pairs:
        pairs = re.findall(
            r"SAMPLE INPUT\n(.*?)\n\nSAMPLE OUTPUT\n(.*?)(?=\n\nSAMPLE INPUT\n|\Z)",
            tail,
            re.DOTALL | re.IGNORECASE,
        )
    if not pairs:
        return None

    inputs = [inp.strip("\n") + "\n" for inp, _ in pairs if inp.strip()]
    outputs = [out.strip("\n") + "\n" for _, out in pairs if out.strip()]
    if not inputs or len(inputs) != len(outputs):
        return None
    return inputs, outputs


def _verify_against_own_response(response: str, ground_truth: str) -> bool:
    """Check whether `response` (the dataset's own expert solution) passes the extracted
    `ground_truth` test cases, using the real `reward_score/code.py` sandboxed runner.

    Only used when `--verify-code-examples` is passed -- it's the same check this
    script's author ran offline to validate the extraction logic in the first place
    (see the module docstring), just applied per-row as an optional, slower, stricter
    quality filter instead of a one-time sanity check.
    """
    from recipe.rgopd.reward_score import code as code_scorer

    result = code_scorer.compute_score(response, ground_truth, extra_info={"split": "train"}, sparse_rewards=False)
    return result["score"] == 1.0


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            (
                "prompt",
                pa.large_list(
                    pa.struct(
                        [
                            ("content", pa.large_string()),
                            ("role", pa.large_string()),
                        ]
                    )
                ),
            ),
            ("data_source", pa.large_string()),
            ("ability", pa.large_string()),
            (
                "reward_model",
                pa.struct(
                    [
                        ("ground_truth", pa.large_string()),
                        ("style", pa.large_string()),
                    ]
                ),
            ),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.large_string()),
                        ("split", pa.large_string()),
                    ]
                ),
            ),
        ]
    )


def convert_row(row: dict, verify_code_examples: bool = False) -> tuple[Optional[dict], Optional[str]]:
    """Convert one UltraInteract_sft row to this repo's schema.

    Returns `(converted_row_or_None, drop_reason_or_None)`: `drop_reason` is a short
    machine-readable tag (used for the per-task summary printed at the end) when the row
    is dropped, `None` when it's kept.
    """
    task = row["task"]
    data_source = TASK_TO_DATA_SOURCE[task]

    if data_source == "math":
        ground_truth = _extract_boxed_answer(row["response"])
        if ground_truth is None:
            return None, "no_boxed_answer"
    else:  # data_source == "code"
        # Only codecontest's embedded example blocks are safe to extract -- see the
        # module docstring's "Ground truth availability" section for why TACO /
        # magicoder_evol_instruct / wiki_table_questions are excluded outright.
        if row["dataset"] != "codecontest":
            return None, "no_verifiable_ground_truth"
        examples = _extract_stdio_examples(row["instruction"])
        if examples is None:
            return None, "example_parse_failed"
        inputs, outputs = examples
        ground_truth = json.dumps(
            {"testtype": "stdin", "fn_name": None, "inputs": inputs, "outputs": outputs, "time_limit": CODECONTEST_TIME_LIMIT_S}
        )
        if verify_code_examples and not _verify_against_own_response(row["response"], ground_truth):
            return None, "failed_self_verification"

    return {
        "prompt": [{"content": row["instruction"], "role": "user"}],
        "data_source": data_source,
        "ability": TASK_TO_ABILITY[task],
        "reward_model": {
            "ground_truth": ground_truth,
            "style": data_source,
        },
        "extra_info": {
            "index": row["id"] if row["id"] else str(uuid.uuid4()),
            "split": "",
        },
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the RG-OPD training subset of openbmb/UltraInteract_sft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for train.parquet/test.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Fraction of kept rows held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=(
            "Comma-separated subset of UltraInteract_sft's `task` values to include "
            f"(default: {DEFAULT_TASKS!r}). `Logic` is accepted syntactically but always "
            "refused -- see the module docstring."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only load this many rows from the source dataset (for a quick smoke test)",
    )
    parser.add_argument(
        "--verify-code-examples",
        action="store_true",
        help=(
            "For codecontest Coding rows, re-run the dataset's own expert response against "
            "its extracted example test cases (via the real reward_score/code.py sandboxed "
            "runner) and drop any row it doesn't pass, instead of only checking that the "
            "examples parsed. Produces a cleaner but smaller kept set (see the module "
            "docstring: ~72.5%% of parsed rows pass this check on a random sample). Slower -- "
            "spawns a sandboxed subprocess per test case for every codecontest row."
        ),
    )
    args = parser.parse_args()

    tasks = _parse_tasks_arg(args.tasks)
    output_dir = Path(args.output_dir)

    print("Downloading openbmb/UltraInteract_sft (train) from HuggingFace...")
    ds = load_dataset("openbmb/UltraInteract_sft", split="train")
    if args.max_samples is not None:
        # UltraInteract_sft is stored as contiguous per-task blocks (Coding, then
        # Math_PoT, then Math_CoT, then Logic), so a plain `train[:N]` prefix -- what
        # this used to do -- silently returns only the first task's rows for any N
        # smaller than that task's block size (114,826 rows as of the version this was
        # tested against), which is almost never what "give me a quick, representative
        # sample" should mean. `datasets` already materializes the full split locally
        # before any slicing regardless (confirmed empirically: the download/convert
        # progress bar always runs over the full row count, `--max-samples` or not), so
        # shuffling first costs nothing extra and actually gives a representative sample.
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
    n_total = len(ds)
    print(f"Loaded {n_total} rows")

    print(f"Converting to project schema (tasks={tasks}, verify_code_examples={args.verify_code_examples})...")
    kept_by_task = {t: 0 for t in tasks}
    dropped_by_task = {t: 0 for t in tasks}
    drop_reasons_by_task: dict[str, dict[str, int]] = {t: {} for t in tasks}
    skipped_other_task = 0
    converted = []

    for row in ds:
        task = row["task"]
        if task not in tasks:
            skipped_other_task += 1
            continue
        out_row, drop_reason = convert_row(row, verify_code_examples=args.verify_code_examples)
        if out_row is None:
            dropped_by_task[task] += 1
            drop_reasons_by_task[task][drop_reason] = drop_reasons_by_task[task].get(drop_reason, 0) + 1
        else:
            kept_by_task[task] += 1
            converted.append(out_row)

    n_kept = len(converted)
    if n_kept < 2:
        # Need at least 1 row on each side of the split; n_kept==0 is the "nothing
        # survived filtering" case, n_kept==1 would otherwise silently produce an empty
        # train.parquet (n_val=max(1,...) always takes the one row) with a misleading
        # "Split: 0 train, 1 val" success message instead of failing loudly.
        raise SystemExit(
            f"Only {n_kept} row(s) survived task filtering + ground-truth extraction out of "
            f"{n_total} loaded (tasks={tasks}); need at least 2 to form a non-empty train/val "
            "split. See the module docstring for why Coding/Math_PoT rows are frequently "
            "dropped, or increase --max-samples."
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_kept)

    n_val = max(1, int(n_kept * args.val_ratio))
    n_train = n_kept - n_val

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    print(f"Split: {n_train} train, {n_val} val (ratio={args.val_ratio})")

    schema = _build_schema()

    def make_table(split_indices, split_name):
        rows = [converted[i] for i in split_indices]
        for row in rows:
            row["extra_info"]["split"] = split_name
        data = {col: [r[col] for r in rows] for col in schema.names}
        return pa.Table.from_pydict(data, schema=schema)

    train_table = make_table(train_indices, "train")
    # Labeled "test" (not "val") to match verl's convention -- the held-out file is named
    # test.parquet, and reward_score/code.py's compute_score checks extra_info["split"]=="test"
    # to switch to full (non-sparse) test-case rewards at validation time.
    val_table = make_table(val_indices, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"

    pq.write_table(train_table, str(train_path))
    pq.write_table(val_table, str(val_path))

    print(f"\nOutput:")
    print(f"  {train_path} ({train_table.num_rows} rows)")
    print(f"  {val_path} ({val_table.num_rows} rows)")

    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, f"BUG: {len(overlap)} overlapping indices"
    print(f"Verification: 0 overlapping samples between train and val")

    print("\nPer-task summary (kept / dropped for missing ground truth):")
    for t in tasks:
        print(f"  {t}: {kept_by_task[t]} kept / {dropped_by_task[t]} dropped")
        for reason, count in sorted(drop_reasons_by_task[t].items(), key=lambda kv: -kv[1]):
            print(f"      - {reason}: {count}")
    if skipped_other_task:
        print(f"  (skipped {skipped_other_task} rows whose task was not in --tasks)")
    for t in tasks:
        if kept_by_task[t] == 0:
            print(
                f"  NOTE: {t} contributed 0 rows -- see the module docstring's "
                "'Ground truth availability' section for why."
            )


if __name__ == "__main__":
    main()
