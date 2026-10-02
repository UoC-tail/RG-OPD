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
"""Build a verified, math-only RG-OPD train/val set from `openbmb/UltraInteract_sft`.

Why a separate math-only prep
------------------------------------------------------------------------------------
`prepare_ultra_interact.py` converts rows one at a time. For `Math_CoT` that parses
100% of rows, but measured on the full dataset the output has two problems that
matter for RL:

  * Prompt duplication. UltraInteract_sft is a set of correction trees: every row is a
    node, and many nodes share one problem statement. The 78,349 Math_CoT rows contain
    only 22,927 distinct problems (3.4x duplication, up to 55 copies of one problem).
    Duplicate prompts over-weight some problems and leak between train and val when the
    split is taken over rows.
  * Contradictory ground truth. The answer is harvested from each row's *response*, and
    some responses in a correction tree are failed attempts. Of the problems with more
    than one row, 602 have answers that the recipe's own math verifier judges
    inequivalent (e.g. `2` vs `200`, `0.15` vs `15`, `1600000` vs `1600005`, and one
    answer that is literally `119 // Incorrectly assumed that ...`). That is ~4.9% of
    Math_CoT rows carrying an answer contradicted by a sibling.

This script works per *problem* instead of per row:

  1. Keep `Math_CoT` rows and strip the fixed two-line instruction wrapper that every
     row shares, keeping the problem statement.
  2. Extract each row's final `\\boxed{...}` answer.
  3. Group rows by problem, and cluster the group's answers by equivalence under
     `recipe/rgopd/reward_score/math.py`'s `verify` -- the same check used as the
     training reward, so "agree" here means "would earn the same reward".
  4. For `gsm8k` and `MATH` problems, match the problem back to its official dataset
     (`openai/gsm8k`, `EleutherAI/hendrycks_math`) and use the official answer. Checked
     against those sources, harvested answers are 100% correct for gsm8k but only ~96.8%
     "verifiably" correct for MATH -- and every mismatch inspected was a correct answer
     left unsimplified (`2(400-100\\pi)+400` for `1200-200\\pi`, `(A)` for `\\text{(A)}`)
     that the verifier cannot prove equal. With the harvested answer as ground truth, a
     student answering in simplified form would get zero reward. Problems from the
     official *test* splits are dropped: lm-eval's `gsm8k` and `minerva_math` score on
     them. (No UltraInteract problem was found in either test split.)
  5. For `mathqa` and `numglue`, which have no usable official source, resolve with
     `--conflict-policy`:
       - `drop` (default): keep only problems whose rows all agree.
       - `majority`: keep the largest answer cluster if it holds a strict majority.
     and drop answers that are not usable as ground truth (annotation text, over-long).
  6. Self-verification: every kept answer must earn reward 1.0 when fed back through
     the training reward as `\\boxed{answer}`. This only catches answers the verifier
     cannot parse; correctness is what step 4 checks.

Output uses the same schema and math system prompt as `prepare_eurus.py`, so the two
datasets are interchangeable in `rgopd_trainer.yaml` (`data.train_files`).

Usage (from the `verl/` directory)
------------------------------------------------------------------------------------
    python -m recipe.rgopd.data.prepare_ultra_interact_math \\
        --output-dir ./data/rgopd-ui-math --n-train 8000 --n-val 200
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from recipe.rgopd.data.prepare_eurus import SYSTEM_PROMPT_MATH, _build_schema
from recipe.rgopd.reward_score import math as math_verifier

HF_REPO = "openbmb/UltraInteract_sft"
HF_FILE = "0000_sft.parquet"

# Official sources for two of the four subsets. UltraInteract's MATH and gsm8k problems are
# the original problem statements, so they can be matched back to their official answers.
GSM8K_REPO = "openai/gsm8k"
MATH_REPO = "EleutherAI/hendrycks_math"

# The two lines every Math_CoT instruction starts with (identical across all 78,349 rows).
# The system prompt already asks for a boxed answer, so the problem is kept on its own.
WRAPPER = (
    "Solve the following math problem step-by-step.\n"
    "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
)

SUBSETS = ("MATH", "gsm8k", "mathqa", "numglue")

# Longer boxed contents are explanations or derivations, not answers.
MAX_ANSWER_CHARS = 200
# A trailing "// ..." or a sentence inside the box is a model's commentary, not an answer.
ANNOTATION_RE = re.compile(r"//|\b(incorrect|assum|because|therefore|however|wrong)\w*\b", re.IGNORECASE)


def extract_boxed(text: str) -> Optional[str]:
    """Content of the last `\\boxed{...}`, brace-matched; None if absent or unbalanced."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : i]
        i += 1
    return None


def unusable_reason(answer: str) -> Optional[str]:
    if not answer.strip():
        return "empty_answer"
    if len(answer) > MAX_ANSWER_CHARS:
        return "answer_too_long"
    if ANNOTATION_RE.search(answer):
        return "annotated_answer"
    return None


def equivalent(a: str, b: str) -> bool:
    """Would a rollout answering `a` earn full reward against ground truth `b`, and vice versa?

    Checked both ways because symbolic comparison is not guaranteed to be symmetric.
    """
    if a == b:
        return True
    return math_verifier.verify(f"\\boxed{{{a}}}", b)[0] and math_verifier.verify(f"\\boxed{{{b}}}", a)[0]


def cluster_answers(answers: list[str]) -> list[list[str]]:
    """Group answers into equivalence clusters, comparing each distinct string only once."""
    counts = Counter(answers)
    clusters: list[tuple[str, list[str]]] = []  # (representative, members)
    for ans, _ in counts.most_common():
        for rep, members in clusters:
            if equivalent(ans, rep):
                members.extend([ans] * counts[ans])
                break
        else:
            clusters.append((ans, [ans] * counts[ans]))
    return sorted((m for _, m in clusters), key=len, reverse=True)


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def load_official_answers() -> dict[tuple[str, str], tuple[str, str]]:
    """(subset, normalized question) -> (official answer, official split).

    GSM8K answers are the number after `####`; MATH answers are the last `\\boxed{}` of the
    reference solution. The split is kept so problems from the *test* splits -- which
    lm-eval's `gsm8k` and `minerva_math` score on -- can be excluded from training.
    """
    from huggingface_hub import snapshot_download

    official: dict[tuple[str, str], tuple[str, str]] = {}
    g = Path(snapshot_download(GSM8K_REPO, repo_type="dataset", allow_patterns=["main/*.parquet"]))
    for f in sorted(g.glob("main/*.parquet")):
        split = "test" if f.name.startswith("test") else "train"
        t = pq.read_table(f)
        for q, a in zip(t["question"].to_pylist(), t["answer"].to_pylist()):
            official[("gsm8k", _norm_question(q))] = (a.split("####")[-1].strip().replace(",", ""), split)
    m = Path(snapshot_download(MATH_REPO, repo_type="dataset", allow_patterns=["*/*.parquet"]))
    for f in sorted(m.glob("*/*.parquet")):
        split = "test" if f.name.startswith("test") else "train"
        t = pq.read_table(f)
        for q, s in zip(t["problem"].to_pylist(), t["solution"].to_pylist()):
            a = extract_boxed(s)
            if a is not None:
                official[("MATH", _norm_question(q))] = (a.strip(), split)
    return official


def canonical(cluster: list[str]) -> str:
    """Most frequent spelling in the cluster; ties go to the shortest (fewest units/words)."""
    counts = Counter(cluster)
    return min(counts, key=lambda a: (-counts[a], len(a)))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare a verified math-only RG-OPD set from openbmb/UltraInteract_sft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=8000, help="Distinct problems in train (0 = all)")
    p.add_argument("--n-val", type=int, default=200, help="Distinct problems in val")
    p.add_argument("--datasets", default=",".join(SUBSETS), help=f"Subset of {SUBSETS} to include")
    p.add_argument("--conflict-policy", choices=("drop", "majority"), default="drop")
    p.add_argument(
        "--no-source-answers",
        action="store_true",
        help="Use harvested answers even where an official GSM8K/MATH answer exists (also disables the test-split guard)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    unknown = wanted - set(SUBSETS)
    if unknown:
        raise SystemExit(f"--datasets: unknown {sorted(unknown)}; choose from {SUBSETS}")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    tbl = pq.read_table(path, columns=["task", "dataset", "instruction", "response"])
    task, subset = tbl["task"].to_pylist(), tbl["dataset"].to_pylist()
    instruction, response = tbl["instruction"].to_pylist(), tbl["response"].to_pylist()

    # ---- 1-2: per-row extraction ------------------------------------------------------
    drops: Counter = Counter()
    problems: dict[str, dict] = {}
    n_rows = 0
    for i in range(len(task)):
        if task[i] != "Math_CoT" or subset[i] not in wanted:
            continue
        n_rows += 1
        if not instruction[i].startswith(WRAPPER):
            drops["row:unexpected_wrapper"] += 1
            continue
        question = instruction[i][len(WRAPPER) :].strip()
        answer = extract_boxed(response[i])
        if answer is None:
            drops["row:no_boxed_answer"] += 1
            continue
        answer = answer.strip()
        entry = problems.setdefault(question, {"subset": subset[i], "answers": []})
        entry["answers"].append(answer)

    # ---- 3-6: per-problem resolution and verification --------------------------------
    official = {} if args.no_source_answers else load_official_answers()

    kept: list[dict] = []
    gt_source: Counter = Counter()
    n_multi = n_conflict = 0
    for question, entry in problems.items():
        answers = entry["answers"]
        if len(answers) > 1:
            n_multi += 1
        match = official.get((entry["subset"], _norm_question(question)))
        if match is not None:
            answer, split = match
            if split == "test":
                # lm-eval's gsm8k / minerva_math score on these splits; training on them
                # would contaminate every reported number.
                drops["problem:in_eval_test_split"] += 1
                continue
            # The official answer is canonical and simplified. A harvested answer can be
            # correct yet unsimplified (`2(400-100\pi)+400` for `1200-200\pi`), which the
            # verifier cannot prove equal -- so a student answering in simplified form would
            # get zero reward. It also settles conflicts between sibling rows.
            source = "official"
        else:
            clusters = cluster_answers(answers)
            if len(clusters) > 1:
                n_conflict += 1
                if args.conflict_policy == "drop":
                    drops["problem:conflicting_answers"] += 1
                    continue
                if len(clusters[0]) * 2 <= len(answers):
                    drops["problem:no_majority_answer"] += 1
                    continue
            answer = canonical(clusters[0])
            reason = unusable_reason(answer)
            if reason:
                drops[f"problem:{reason}"] += 1
                continue
            source = "harvested"
        if len(answer) > MAX_ANSWER_CHARS:
            drops["problem:answer_too_long"] += 1
            continue
        if math_verifier.compute_score(f"\\boxed{{{answer}}}", answer)["score"] != 1.0:
            drops["problem:fails_self_verification"] += 1
            continue
        gt_source[source] += 1
        kept.append({"question": question, "answer": answer, "subset": entry["subset"], "n_rows": len(answers)})

    # ---- split over distinct problems, so train and val cannot share a problem --------
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = min(args.n_val, len(kept))
    val_items = kept[:n_val]
    rest = kept[n_val:]
    train_items = rest if args.n_train <= 0 else rest[: args.n_train]
    if len(train_items) < args.n_train:
        print(f"warning: only {len(train_items)} verified problems available for train (asked {args.n_train})")

    schema = _build_schema()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for items, split, fname in ((train_items, "train", "train.parquet"), (val_items, "test", "test.parquet")):
        rows = [
            {
                "prompt": [
                    {"content": SYSTEM_PROMPT_MATH, "role": "system"},
                    {"content": it["question"], "role": "user"},
                ],
                "data_source": "math",
                "ability": "math",
                "reward_model": {"ground_truth": it["answer"], "style": "math"},
                "extra_info": {"index": f"ultrainteract/{it['subset']}", "split": split},
            }
            for it in items
        ]
        table = pa.Table.from_pydict({c: [r[c] for r in rows] for c in schema.names}, schema=schema)
        pq.write_table(table, str(out_dir / fname))

    # ---- report ------------------------------------------------------------------------
    print(f"Math_CoT rows considered : {n_rows}")
    print(f"distinct problems        : {len(problems)}  ({n_rows / max(len(problems), 1):.2f} rows per problem)")
    print(f"problems with >1 row     : {n_multi}")
    print(f"  ...with conflicting answers: {n_conflict}  (policy={args.conflict_policy})")
    print(f"verified problems kept   : {len(kept)}")
    for reason, count in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {reason}: {count}")
    print("ground truth source      : " + ", ".join(f"{k}={v}" for k, v in gt_source.most_common()))
    by_subset = Counter(it["subset"] for it in kept)
    print("kept by subset           : " + ", ".join(f"{k}={v}" for k, v in by_subset.most_common()))
    print(f"wrote {out_dir / 'train.parquet'} ({len(train_items)}) and {out_dir / 'test.parquet'} ({len(val_items)})")


if __name__ == "__main__":
    main()
