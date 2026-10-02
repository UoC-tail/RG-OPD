#!/usr/bin/env python3
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
"""Build an RG-OPD train/val subset from `PRIME-RL/Eurus-2-RL-Data`.

Why this exists alongside `prepare_ultra_interact.py`
------------------------------------------------------------------------------------
`prepare_ultra_interact.py` has to *scrape* ground truth out of free text, because
`openbmb/UltraInteract_sft` ships no machine-checkable answer field: math answers are
harvested from the expert response's `\\boxed{...}`, and code test cases are parsed out
of the "Examples/Input/Output" block of the problem statement. Both are lossy.

`PRIME-RL/Eurus-2-RL-Data` (the dataset PRIME itself trains on --  see
`PRIME/training/scripts/data_prepare.py`, which does no parsing at all) already carries
real ground truth and is already in verl's schema. Measured differences on this repo's
own verifiers:

  * Code test cases are the problems' *real* test suites (median 7 for `taco`, 101 for
    `codecontests`), not the 1-3 illustrative examples embedded in a problem statement.
  * Every code prompt is unique (25,276 of them). The UltraInteract path yields ~4.7k
    usable unique coding problems at 6.4x row duplication, because every row is a node
    in a correction tree and only the instruction is kept.
  * Math ground truth is a real reference answer, not an answer harvested from a
    response -- ~4.9% of `Math_CoT` rows scraped from UltraInteract carry an answer
    that a sibling row for the same prompt contradicts (i.e. it came from a failed
    attempt inside the correction tree).

Two conversions are still needed, and this script does them:

1. `data_source` remap. Eurus labels rows by origin (`taco`, `codecontests`, `apps`,
   `codeforces`, `numina_*`); `recipe/rgopd/reward_score/__init__.py` dispatches on
   `"code"` / `"math"`. Rows are relabeled accordingly (origin is preserved in
   `extra_info.index`).
2. Code ground-truth shim. Eurus stores `{"inputs", "outputs"[, "fn_name"]}` with the
   two lists *repr'd as Python source strings*; `reward_score/code.py` wants a parsed
   `{"testtype", "fn_name", "inputs", "outputs", "time_limit"}`. See `_convert_code_gt`.

Test-case capping (`--max-test-cases`, important)
------------------------------------------------------------------------------------
`reward_score/code.py` spawns **one sandboxed subprocess per test case, all at once**.
`reward_score/__init__.py` caps this at 20 during training, but `code.py` deliberately
ignores that cap whenever `extra_info["split"] == "test"`, so validation would run a
problem's *entire* suite -- for `codecontests` that is a median of 101 processes per
response, times `rollout.val_kwargs.n` responses, times the whole val set. This script
therefore truncates each problem's suite at prep time (default 12) so validation stays
tractable. Raise it if you have the CPU budget; note it bounds reward fidelity, not
just cost.

Functional (`fn_name`) problems
------------------------------------------------------------------------------------
~10% of Eurus code rows (2,658: all of `taco`'s function-signature problems plus some
`apps`) are functional rather than stdin: the test case is a *positional argument list*
(`[[4, 4, 4, 3, 3], 12]` meaning `fn(*args)`), and the expected value is usually
wrapped in a 1-element list. `reward_score/code.py`'s `run_test_func` accepts only a
dict (splatted as `**kwargs`) or a whitespace-separated string of JSON tokens, so it
cannot consume that format. Those rows are **dropped by default**; `--include-functional`
keeps them but will score them as failures until `run_test_func` grows a positional-args
path. See `recipe/rgopd/README.md`.

Usage
------------------------------------------------------------------------------------
    python -m recipe.rgopd.data.prepare_eurus --output-dir ./data/rgopd-eurus \\
        --n-train 8000 --n-val 200

Run from the `verl/` directory, like the rest of this recipe.
"""

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

HF_REPO = "PRIME-RL/Eurus-2-RL-Data"

# Per-test-case timeout (seconds) handed to code.py as test_cases["time_limit"].
DEFAULT_TIME_LIMIT_S = 3.0

# Eurus `data_source` -> the two values reward_score/__init__.py dispatches on.
CODE_SOURCES = {"taco", "codecontests", "apps", "codeforces"}


# Eurus prompts ship with PRIME's own [ASSESS]/[ADVANCE]/... action-format system
# prompt, which is specific to PRIME's method and would push the student into that
# output format. We replace it with the minimal instruction this recipe's verifiers
# actually require: reward_score/math.py scores only a `\boxed{...}` answer, and
# reward_score/code.py's `extract_code` returns None (-> "Incorrect format", reward 0)
# unless the response contains a ``` fenced block.
SYSTEM_PROMPT_MATH = (
    "You are a careful mathematical reasoner. Think step by step, then give the final "
    "answer inside \\boxed{}."
)
SYSTEM_PROMPT_CODE = (
    "You are an expert Python programmer. Think step by step, then give the complete "
    "program inside a single ```python code block. The program must read from standard "
    "input and write to standard output."
)


def _convert_code_gt(raw: str, max_test_cases: int, include_functional: bool) -> tuple[Optional[str], Optional[str]]:
    """Eurus `reward_model.ground_truth` -> reward_score/code.py's test-case schema.

    Returns `(ground_truth_json, drop_reason)`; exactly one is non-None.
    """
    obj = json.loads(raw)
    inputs, outputs = obj["inputs"], obj["outputs"]
    # Eurus repr's both lists as Python source, e.g. "['3 3\\n1 2 1\\n', '2 2\\n']".
    if isinstance(inputs, str):
        inputs = ast.literal_eval(inputs)
    if isinstance(outputs, str):
        outputs = ast.literal_eval(outputs)
    if not inputs or len(inputs) != len(outputs):
        return None, "malformed_test_cases"

    fn_name = obj.get("fn_name")
    if fn_name and not include_functional:
        return None, "functional_unsupported"

    inputs = inputs[:max_test_cases]
    outputs = outputs[:max_test_cases]

    if not fn_name:
        # stdin: code.py feeds each input to sys.stdin and compares trimmed stdout.
        # A few sources store an input/output as a list of lines rather than one blob.
        inputs = [x if isinstance(x, str) else "\n".join(map(str, x)) for x in inputs]
        outputs = [x if isinstance(x, str) else "\n".join(map(str, x)) for x in outputs]

    return json.dumps(
        {
            "testtype": "functional" if fn_name else "stdin",
            "fn_name": fn_name,
            "inputs": inputs,
            "outputs": outputs,
            "time_limit": DEFAULT_TIME_LIMIT_S,
        }
    ), None


def _build_schema() -> pa.Schema:
    """Identical to prepare_ultra_interact.py's schema, so both preps are drop-in swaps."""
    return pa.schema(
        [
            ("prompt", pa.large_list(pa.struct([("content", pa.large_string()), ("role", pa.large_string())]))),
            ("data_source", pa.large_string()),
            ("ability", pa.large_string()),
            ("reward_model", pa.struct([("ground_truth", pa.large_string()), ("style", pa.large_string())])),
            ("extra_info", pa.struct([("index", pa.large_string()), ("split", pa.large_string())])),
        ]
    )


def convert_row(source: str, ability: str, prompt, raw_gt: str, args) -> tuple[Optional[dict], Optional[str]]:
    is_code = source in CODE_SOURCES or ability == "code"
    if is_code:
        ground_truth, reason = _convert_code_gt(raw_gt, args.max_test_cases, args.include_functional)
        if ground_truth is None:
            return None, reason
        data_source = "code"
        system_prompt = SYSTEM_PROMPT_CODE
    else:
        ground_truth = raw_gt
        if not ground_truth.strip():
            return None, "empty_ground_truth"
        data_source = "math"
        system_prompt = SYSTEM_PROMPT_MATH

    # Drop PRIME's action-format system turn, keep the problem statement.
    user_turns = [m for m in prompt if m["role"] == "user"]
    if not user_turns:
        return None, "no_user_turn"

    return {
        "prompt": [
            {"content": system_prompt, "role": "system"},
            {"content": user_turns[-1]["content"], "role": "user"},
        ],
        "data_source": data_source,
        "ability": "code" if is_code else "math",
        "reward_model": {"ground_truth": ground_truth, "style": data_source},
        "extra_info": {"index": source, "split": ""},
    }, None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare an RG-OPD subset from PRIME-RL/Eurus-2-RL-Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=8000, help="Number of training rows to keep")
    p.add_argument("--n-val", type=int, default=200, help="Number of validation rows to keep")
    p.add_argument("--code-frac", type=float, default=0.5, help="Fraction of kept rows that should be code")
    p.add_argument("--max-test-cases", type=int, default=12, help="Truncate each code problem's suite to this many")
    p.add_argument("--include-functional", action="store_true", help="Keep fn_name problems code.py cannot run yet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Fetching {HF_REPO} ...")
    paths = {s: hf_hub_download(HF_REPO, f"{s}.parquet", repo_type="dataset") for s in ("train", "validation")}

    schema = _build_schema()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for split_in, split_out, n_want, fname in (
        ("train", "train", args.n_train, "train.parquet"),
        ("validation", "test", args.n_val, "test.parquet"),
    ):
        tbl = pq.read_table(paths[split_in], columns=["data_source", "ability", "prompt", "reward_model"])
        src = tbl["data_source"].to_pylist()
        abil = tbl["ability"].to_pylist()
        prompts = tbl["prompt"].to_pylist()
        rms = tbl["reward_model"].to_pylist()

        idx = list(range(len(src)))
        rng.shuffle(idx)

        n_code_want = int(n_want * args.code_frac)
        n_math_want = n_want - n_code_want
        code_rows, math_rows, drops = [], [], {}
        for i in idx:
            if len(code_rows) >= n_code_want and len(math_rows) >= n_math_want:
                break
            is_code = src[i] in CODE_SOURCES or abil[i] == "code"
            if is_code and len(code_rows) >= n_code_want:
                continue
            if not is_code and len(math_rows) >= n_math_want:
                continue
            row, reason = convert_row(src[i], abil[i], prompts[i], rms[i]["ground_truth"], args)
            if row is None:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            row["extra_info"]["split"] = split_out
            (code_rows if is_code else math_rows).append(row)

        rows = code_rows + math_rows
        rng.shuffle(rows)
        table = pa.Table.from_pydict({c: [r[c] for r in rows] for c in schema.names}, schema=schema)
        path = out_dir / fname
        pq.write_table(table, str(path))
        print(f"  {path}: {table.num_rows} rows ({len(code_rows)} code / {len(math_rows)} math)")
        if drops:
            for reason, count in sorted(drops.items(), key=lambda kv: -kv[1]):
                print(f"      dropped {reason}: {count}")


if __name__ == "__main__":
    main()
