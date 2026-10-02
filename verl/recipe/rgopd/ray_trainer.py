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
"""RG-OPD trainer: verl's stock `RayPPOTrainer` (`fit()`, rollout, advantage computation,
checkpointing, validation, ...) is reused unchanged. The only thing RG-OPD needs is,
when `teacher.source="external"`, to query the external teacher server for top-K
log-probs on the just-sampled rollout batch and attach them before the actor update --
so this trainer overrides only `_update_actor`, the single seam `RayPPOTrainer.fit()`
already calls out to for that step (see `verl/verl/trainer/ppo/ray_trainer.py`).

For `teacher.source in {"ema", "trust-region"}` `_update_actor` is a no-op passthrough:
the self-teacher forward happens inside `recipe/rgopd/dp_actor.py` on the same rollout
batch the actor already has, using `self.teacher_module` directly (no separate worker-
group RPC). Those two teacher sources DO need one thing from this trainer, though:
`main_rgopd.py` colocates the reference model onto `Role.ActorRolloutRef` so that
`self.teacher_module` exists at all, and verl's `need_reference_policy()` keys
*only* off role-mapping presence -- so merely being colocated makes
`RayPPOTrainer.fit()` unconditionally run a full reference-model forward pass every
step (`_compute_ref_log_prob`) regardless of `use_kl_in_reward`/`use_kl_loss` (both
always False here). RG-OPD never reads that `ref_log_prob` off the batch, so
`_compute_ref_log_prob` is overridden below to skip it.
"""

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .core_algos import _cfg

__all__ = ["RGOPDRayTrainer"]


class RGOPDRayTrainer(RayPPOTrainer):
    def _rgopd_config(self):
        rgopd_cfg = self.config.actor_rollout_ref.actor.get("recipe", {}).get("rgopd", None)
        if rgopd_cfg is None:
            raise ValueError(
                "RGOPDRayTrainer requires actor_rollout_ref.actor.recipe.rgopd config "
                "(see recipe/rgopd/config/rgopd_trainer.yaml)."
            )
        return rgopd_cfg

    def _external_teacher_client(self):
        if not hasattr(self, "_rgopd_teacher_client_cache"):
            teacher_cfg = _cfg(self._rgopd_config(), "teacher", {})
            if _cfg(teacher_cfg, "source", "external") != "external":
                self._rgopd_teacher_client_cache = None
            else:
                from recipe.gkd.teacher import TeacherClient

                ext_cfg = _cfg(teacher_cfg, "external", {})
                n_workers = int(_cfg(ext_cfg, "n_server_workers", 1))
                client = TeacherClient(
                    server_ip=_cfg(ext_cfg, "server_ip", "localhost"),
                    server_port=int(_cfg(ext_cfg, "server_port", 15555)),
                    n_server_workers=n_workers,
                    max_tokens=1,
                    only_response=False,
                )
                print(
                    f"[RG-OPD] external teacher client -> "
                    f"{_cfg(ext_cfg, 'server_ip', 'localhost')}:{_cfg(ext_cfg, 'server_port', 15555)} "
                    f"(n_server_workers={n_workers})"
                )
                self._rgopd_teacher_client_cache = (client, n_workers)
        return self._rgopd_teacher_client_cache

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # See the module docstring: this is called unconditionally by RayPPOTrainer.fit()
        # whenever a reference model is colocated (teacher.source in {ema, trust-region}),
        # but RG-OPD never consumes its result. `fit()` immediately does
        # `batch.union(ref_log_prob)`, which requires matching batch_size (DataProto.
        # from_dict(tensors={}) leaves batch_size=None and fails that check) -- so return
        # a same-batch-size, zero-field TensorDict, making the union a no-op instead of
        # paying for a real reference-model forward pass every step.
        return DataProto(batch=TensorDict({}, batch_size=batch.batch.batch_size))

    def _update_actor(self, batch: DataProto) -> DataProto:
        cached = self._external_teacher_client()
        if cached is not None:
            client, n_workers = cached
            from recipe.gkd.teacher_utils import get_teacher_knowledge

            teacher_knowledge = get_teacher_knowledge(batch, client, n_server_workers=n_workers, is_async=False)
            ext_data = DataProto.from_dict(
                tensors={
                    "ext_teacher_topk_logps": torch.from_numpy(teacher_knowledge.non_tensor_batch["teacher_topk_logps"]),
                    "ext_teacher_topk_indices": torch.from_numpy(
                        teacher_knowledge.non_tensor_batch["teacher_topk_indices"]
                    ).long(),
                }
            )
            batch = batch.union(ext_data)
            batch.meta_info["use_external_teacher"] = True

        return super()._update_actor(batch)
