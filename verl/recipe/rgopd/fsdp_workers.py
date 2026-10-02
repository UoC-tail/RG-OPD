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
"""FSDP worker for RG-OPD: identical to verl's stock `AsyncActorRolloutRefWorker`, except
the actor is an `RGOPDActor` (recipe/rgopd/dp_actor.py) and, for the self-teacher
(`teacher.source in {"ema", "trust-region"}`) setting, the actor's `teacher_module` is
wired up from the colocated reference model. For `teacher.source="external"` this is a
no-op: no local teacher weights are needed, since `recipe/rgopd/ray_trainer.py` queries
the external teacher server directly.
"""

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .core_algos import _cfg
from .dp_actor import RGOPDActor, TrustRegionTeacher

__all__ = ["RGOPDActorRolloutRefWorker"]


class RGOPDActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Do everything the stock worker does (build/shard the actor, rollout, and --
        # when role="actor_rollout_ref" -- the colocated reference model). This leaves
        # `self.actor` as a base `DataParallelPPOActor`; we then swap it for an
        # `RGOPDActor` below, reusing the exact same underlying module/optimizer (cheap:
        # no new weights are allocated).
        super().init_model()

        if not self._is_actor:
            return

        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = RGOPDActor(
            config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
        )

        rgopd_cfg = actor_cfg.get("recipe", {}).get("rgopd", None)
        if rgopd_cfg is None:
            raise ValueError(
                "RGOPDActorRolloutRefWorker requires actor_rollout_ref.actor.recipe.rgopd config "
                "(see recipe/rgopd/config/rgopd_trainer.yaml)."
            )
        teacher_cfg = _cfg(rgopd_cfg, "teacher", {})
        source = _cfg(teacher_cfg, "source", "external")

        if source == "external":
            return  # no local teacher weights; ray_trainer.py queries the external server.

        if not self._is_ref:
            raise ValueError(
                f"teacher.source='{source}' requires the ref model to be colocated with the actor "
                "(role='actor_rollout_ref'); see recipe/rgopd/main_rgopd.py."
            )
        if source == "ema":
            self.actor.teacher_module = self.ref_module_fsdp
        elif source == "trust-region":
            self.actor.teacher_module = TrustRegionTeacher(
                ref_module=self.ref_module_fsdp,
                student_module=self.actor_module_fsdp,
                mix_coef=float(_cfg(teacher_cfg, "trust_region_mix_coef", 0.0)),
            )
        else:
            raise ValueError(f"Unknown teacher.source: {source!r}. Must be one of: external, ema, trust-region")
