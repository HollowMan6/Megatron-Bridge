# Megatron-Bridge (MindLab private mirror)

This repository is a **private MindLab mirror** of
[NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).
It is NOT a GitHub fork (no upstream PR plumbing); `main` is a plain
mirror of `upstream/main`.

## Sync policy (one-way)

- `main` tracks `upstream/main` (NVIDIA-NeMo) and is updated by
  **fast-forward push only**: `git fetch upstream && git push mindlab
  upstream/main:main`. If local meta commits (this AGENTS.md and files
  like it) sit on main, rebase them onto the new upstream head before
  pushing — never merge, never rebase upstream commits onto ours.
- We **cherry-pick upstream changes we want** into our feature branches
  when a full sync is not yet safe.
- We **never open PRs upstream**. All MindLab work (model support, PEFT
  extensions, fixes) stays in this private repo. Upstreaming is a
  deliberate, separately-approved activity outside normal workflow.

## Internal workflow

- Feature work happens on `feat/*` branches cut from `main`, delivered
  via PRs to this repo's `main` (never pushed to `upstream`).
- Keep the diff against `upstream/main` minimal and additive: new files
  and extension points beat edits to vendored upstream files — it keeps
  future syncs conflict-free.
- Deployment consumers pin this repo by exact rev (see mint
  `recipes/*/megatron/pyproject.toml` `[tool.mint.libs]`); changing a
  pin requires re-running the consumer's verification gates.

## Fork Notes (MindLab-Research)

- `main` carries this fork's work and syncs from `NVIDIA-NeMo/Megatron-Bridge`
  `main` by merge.
- The `all/*` refs belong to a separate contributor's line; they are not part of
  this line and are not merged into it.
- GLM-5 model support lives here (`src/megatron/bridge/models/glm_moe_dsa/`) and
  depends only on `GPTModel` plus
  `megatron/core/models/gpt/experimental_attention_variant_module_specs` from
  Megatron-LM.
- The batched multi-LoRA kernels live in `src/megatron/bridge/peft/batched/`.
