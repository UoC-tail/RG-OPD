# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
"""Verifier for boxed-answer math problems (GSM8K/MATH-style `\\boxed{...}` answers)."""

import re
import signal
from typing import Optional

from math_verify import parse as mv_parse
from math_verify import verify as mv_verify


def normalize_latex(s: str) -> str:
    """Normalize a LaTeX string for robust comparison.

    Handles common formatting differences that are mathematically equivalent:
    - \\dfrac vs \\frac, shorthand \\frac args
    - \\left( / \\right) vs plain ( / )
    - \\$ prefix, \\! spacing, thousand-separator commas
    - Whitespace around operators, commas, and before LaTeX commands
    - \\text{(X)} vs \\textbf{(X)} for multiple-choice
    - ``x \\in`` prefix on intervals
    """
    s = s.strip()
    # \dfrac -> \frac
    s = s.replace(r"\dfrac", r"\frac")
    # \left and \right delimiters
    s = re.sub(r"\\left\s*([(\[{|])", r"\1", s)
    s = re.sub(r"\\right\s*([)\]}|])", r"\1", s)
    s = re.sub(r"\\left\s*\\([{|])", r"\\\1", s)
    s = re.sub(r"\\right\s*\\([}|])", r"\\\1", s)
    # \$ -> empty, \! -> empty (cosmetic LaTeX)
    s = s.replace(r"\$", "")
    s = s.replace(r"\!", "")
    # Remove thousand-separator commas in numbers: 32,348 -> 32348
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    # Strip "x \in " or "x\in " prefix on intervals/sets
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)
    # Expand shorthand \frac: \frac12 -> \frac{1}{2}
    s = re.sub(
        r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])(?![a-zA-Z{])",
        r"\\frac{\1}{\2}",
        s,
    )
    # Expand shorthand \frac where first arg is single char, second is braced:
    # \frac9{19} -> \frac{9}{19}
    s = re.sub(
        r"\\frac\s*([0-9a-zA-Z])\s*(\{)",
        r"\\frac{\1}\2",
        s,
    )
    # Expand shorthand \frac where first arg is braced, second is single char:
    # \frac{270}7 -> \frac{270}{7}
    s = re.sub(
        r"(\\frac\{[^}]*\})\s*([0-9a-zA-Z])(?![a-zA-Z{])",
        r"\1{\2}",
        s,
    )
    # \textbf{...} -> \text{...}
    s = s.replace(r"\textbf", r"\text")
    # Normalize \text{(X)} -> \text{X} for multiple-choice answers
    s = re.sub(r"\\text\{?\(([A-E])\)\}?", r"\\text{\1}", s)
    # Strip \text{...} wrappers so \text{C} and bare C are equivalent
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # Normalize whitespace: collapse runs, strip around punctuation/operators
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s*\+\s*", "+", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*=\s*", "=", s)
    # Remove spaces before backslash-commands (e.g., "5 \pi" -> "5\pi")
    s = re.sub(r"(\d)\s+(\\[a-zA-Z])", r"\1\2", s)
    # Remove spaces inside braces: { x } -> {x}
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    # Remove spaces inside parens/brackets for tuples/intervals
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    return s.strip()


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression (`\\boxed{...}`, brace-matched) from a string."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    r"""Strip the `\boxed{...}` wrapper, returning "" if `s` isn't well-formed."""
    left = r"\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return ""


class timeout:
    """Context manager that raises `TimeoutError` after `seconds` (POSIX `SIGALRM`-based)."""

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def verify(solution_str: str, answer: str) -> tuple[bool, Optional[str]]:
    """Check whether `solution_str`'s boxed answer matches `answer`.

    Tries, in order: exact string match, LaTeX-normalized match, and finally symbolic
    equivalence via `math_verify` (handles e.g. `1/2` vs `0.5`, reordered set/tuple
    elements). Returns `(is_correct, extracted_prediction)`; `extracted_prediction` is
    `None` if no `\\boxed{...}` was found at all.
    """
    boxed_pred = last_boxed_only_string(solution_str)
    pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    correct = pred == answer
    if not correct and pred:
        correct = normalize_latex(pred) == normalize_latex(answer)
    if not correct and pred:
        try:
            with timeout(seconds=5):
                correct = mv_verify(mv_parse(answer), mv_parse(pred))
        except Exception:  # noqa: BLE001 - parsing/verification errors just count as incorrect
            pass
    return correct, pred


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> dict:
    """Reward-score a boxed-answer math response.

    Returns a dict with `score`/`acc` in {0.0, 1.0}, the extracted `pred` (may be `None`
    if no `\\boxed{...}` was found), and `incorrect_format`/`truncated` flags carried
    through for logging (see `recipe/rgopd/README.md#reward-scoring`).
    """
    extra_info = extra_info or {}
    was_truncated = bool(extra_info.get("truncated", False))

    correct, pred = verify(solution_str, ground_truth)
    reward = 1.0 if correct else 0.0
    incorrect_format = pred is None or pred == ""

    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
    }
