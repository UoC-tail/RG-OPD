# RG-OPD: Reward-Gated On-Policy Distillation

Official code for:

> Mohammad Sadegh Akhondzadeh, Vijay Lingam, Atula Tejaswi, Chanakya Ekbote, Sujay Sanghavi,
> Aleksandar Bojchevski. **Reward-Gated On-Policy Distillation.** arXiv:2607.04037, 2026.
> https://arxiv.org/abs/2607.04037

On-policy distillation samples trajectories from the student's own policy and has a fixed
teacher provide dense, token-level supervision on them. That supervision isn't always
trustworthy, though: a teacher can assign high likelihood to a plausible-but-wrong solution,
or low likelihood to a correct one the student phrased differently — and distilling
unconditionally then reinforces exactly those mistakes. **RG-OPD** gates the distillation
loss with a simple rule: trust the teacher on a trajectory only when its likelihood relative
to the student agrees with what the verifier reward already told you about that trajectory.
Across reasoning and coding benchmarks this outperforms both vanilla reverse-KL on-policy
distillation and TSD-KD.

This repository is [verl](https://github.com/volcengine/verl) (Bytedance's RL-for-LLMs
training framework) plus one added recipe, `verl/recipe/rgopd/`, following the same
structure as verl's other recipes (`dapo`, `gkd`, `spin`, ...).

## Layout

```
verl/                      vendored verl framework (see verl/recipe/rgopd/README.md#patches-to-verl
                            for the short list of small changes made to it)
  recipe/rgopd/             <- RG-OPD: the method, training entry point, config, data prep, eval docs
    core_algos.py            the loss (Eq. 1-4 of the paper)
    dp_actor.py               actor: top-K distillation forward + the RG-OPD training step
    fsdp_workers.py           worker wiring (self-teacher setup)
    ray_trainer.py            trainer: external-teacher querying
    main_rgopd.py             entry point (python -m recipe.rgopd.main_rgopd)
    config/rgopd_trainer.yaml training config (student/teacher models, distillation/gate settings)
    reward_score/              math + code verifiers used as the training-time reward
    data/prepare_ultra_interact.py  training-data prep
    scripts/                   serve_teacher.sh, train_rgopd.sh
    eval/README.md             the paper's official eval protocol (lm-evaluation-harness)
    README.md                  <- start here for the method, config reference, and full run instructions
```

## Quickstart

```bash
git clone <this repo>
cd RG-OPD/verl
pip install -e .[vllm]   # see verl/README.md for the full/alternative install options

python -m recipe.rgopd.data.prepare_ultra_interact --output-dir ./data/rgopd
bash recipe/rgopd/scripts/serve_teacher.sh &
bash recipe/rgopd/scripts/train_rgopd.sh
```

See **[`verl/recipe/rgopd/README.md`](verl/recipe/rgopd/README.md)** for the method
write-up, the config reference, teacher-source options, and evaluation instructions.

## Citation

```bibtex
@article{akhondzadeh2026rgopd,
  title   = {Reward-Gated On-Policy Distillation},
  author  = {Akhondzadeh, Mohammad Sadegh and Lingam, Vijay and Tejaswi, Atula and Ekbote, Chanakya and Sanghavi, Sujay and Bojchevski, Aleksandar},
  journal = {arXiv preprint arXiv:2607.04037},
  year    = {2026}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). This repository is a
fork of [verl](https://github.com/volcengine/verl) (Copyright 2023-2024 Bytedance Ltd.
and/or its affiliates), also Apache 2.0 licensed.
