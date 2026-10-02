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
"""Rule-based verifier dispatch for the RG-OPD recipe.

Routes a `(data_source, solution_str, ground_truth, extra_info)` reward request to the
matching verifier in this package (`math.compute_score` or `code.compute_score`). This
module is referenced from `custom_reward_function.path` in
`recipe/rgopd/config/rgopd_trainer.yaml`.

It intentionally covers only the math/code subset of UltraInteract that this recipe
trains on by default (see `recipe/rgopd/data/prepare_ultra_interact.py`) — extend the
dispatch table below if you add other data sources.
"""

from . import code
from . import math


# Cap on test cases run per code response during training (code.py spawns one sandboxed
# subprocess per test case, all at once, with no other concurrency limit). code.py itself
# ignores this and runs every test case whenever extra_info["split"]=="test", so full-
# fidelity validation/eval scoring is unaffected -- this only bounds worst-case process
# spawn during training reward computation. The codecontest-derived examples this recipe
# ships (recipe/rgopd/data/prepare_ultra_interact.py) carry only a handful of test cases
# each, so this cap won't bind for them; raise it if you plug in a dataset with more.
_MAX_CODE_TEST_CASES_TRAIN = 20


# verl's validation-metric aggregation (verl/trainer/ppo/ray_trainer.py, the
# `assert len(lst) == 0 or len(lst) == len(sample_scores)` after the val loop) requires
# EVERY sample in a batch to contribute the SAME set of reward_extra_info keys. math.py
# and code.py do not return the same keys -- code.py adds `error_in_test_cases` and
# `timed_out`, which have no meaning for a boxed-answer check -- so any batch mixing
# data_source="math" and data_source="code" rows (i.e. every batch this recipe trains
# on) would otherwise fail that assert as soon as validation runs:
#     AssertionError: error_in_test_cases: len(lst)=400, len(sample_scores)=800
# Normalizing here rather than in each verifier keeps the dispatch table the single
# place that has to know the union schema.
_EXTRA_INFO_DEFAULTS = {
    "acc": 0.0,
    "pred": "",
    "incorrect_format": 0,
    "error_in_test_cases": 0,
    "timed_out": 0,
    "truncated": 0,
    "truncated_and_missing_answer": 0,
}


def _normalize(result: dict) -> dict:
    """Pad a verifier's result to the union key schema described above.

    Also coerces `pred` to a string: math.py returns None when no `\\boxed{...}` was
    found, and a list mixing None with code.py's always-str `pred` breaks downstream
    metric aggregation.
    """
    out = dict(_EXTRA_INFO_DEFAULTS)
    out.update(result)
    out["pred"] = "" if out.get("pred") is None else str(out["pred"])
    return out


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    if data_source in ("math", "math_train", "math500", "gsm8k") or data_source.startswith("aime"):
        return _normalize(math.compute_score(solution_str, ground_truth, extra_info))
    elif data_source in ("code", "livecodebench", "humanevalplus"):
        return _normalize(
            code.compute_score(
                solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=_MAX_CODE_TEST_CASES_TRAIN
            )
        )
    else:
        raise ValueError(
            f"Unrecognized data_source: {data_source!r}. This recipe only ships math and code "
            "verifiers (see reward_score/math.py, reward_score/code.py); add your own dispatch "
            "entry here for other data sources."
        )
