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
"""Entry point for RG-OPD training: `python -m recipe.rgopd.main_rgopd`.

Reuses `verl.trainer.main_ppo`'s `run_ppo()` driver (Ray init, checkpoint download,
tokenizer/dataset/reward-function setup, ...) wholesale via its `task_runner_class` hook
-- the exact extension point verl's own TaskRunner docstring calls out "for recipe to
change TaskRunner" -- and only customizes:
  - `add_actor_rollout_worker`: use `RGOPDActorRolloutRefWorker`, and colocate a
    reference model (`Role.ActorRolloutRef`) only when the self-teacher
    (`teacher.source in {"ema", "trust-region"}`) path needs one.
  - `add_ref_policy_worker`: skip adding a standalone `RefPolicy` worker when the
    reference model was already colocated above (RG-OPD doesn't use the standard
    KL-to-reference penalty, so there's never a reason to add a separate one).
  - `run`: use `RGOPDRayTrainer` in place of `RayPPOTrainer`. `TaskRunner.run()` builds
    the trainer inline rather than through an overridable hook, so this substitutes the
    name in `verl.trainer.main_ppo`'s module namespace for the duration of the call
    instead of duplicating that ~90-line method just to change one line.
"""

from unittest.mock import patch

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.ray_trainer import Role
from verl.utils.device import auto_set_device

from .core_algos import _cfg
from .fsdp_workers import RGOPDActorRolloutRefWorker
from .ray_trainer import RGOPDRayTrainer

__all__ = ["RGOPDTaskRunner", "main"]


class RGOPDTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            raise NotImplementedError(
                "recipe/rgopd does not support trainer.use_legacy_worker_impl=disable (the new "
                "tensordict-based worker engine): RGOPDActor.update_policy takes a DataProto and "
                "does its own batching/micro-batching, matching the legacy engine's calling "
                "convention, not the new engine's. Leave use_legacy_worker_impl at its default."
            )

        rgopd_cfg = config.actor_rollout_ref.actor.get("recipe", {}).get("rgopd", None)
        teacher_cfg = _cfg(rgopd_cfg, "teacher", {}) if rgopd_cfg is not None else {}
        needs_colocated_teacher = _cfg(teacher_cfg, "source", "external") in ("ema", "trust-region")

        role = Role.ActorRolloutRef if needs_colocated_teacher else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(RGOPDActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return RGOPDActorRolloutRefWorker, RayWorkerGroup

    def add_ref_policy_worker(self, config, ref_policy_cls):
        if Role.ActorRolloutRef in self.role_worker_mapping:
            return  # reference model already colocated with the actor
        super().add_ref_policy_worker(config, ref_policy_cls)

    def run(self, config):
        with patch("verl.trainer.main_ppo.RayPPOTrainer", RGOPDRayTrainer):
            super().run(config)


@hydra.main(config_path="config", config_name="rgopd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(RGOPDTaskRunner))


if __name__ == "__main__":
    main()
