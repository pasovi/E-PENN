# E-PENN — game-trained embeddings in pursuit–evasion

Two directories:

| | |
|---|---|
| [`original_paper_reproduction/`](original_paper_reproduction/) | The NumPy code, checkpoints, results and figures behind the draft *"What Does a Game-Trained Embedding Learn? Strategy Ensembles versus Latent Overhead in Pursuit–Evasion"*, exactly as they were on `main` before November 2026 (see its `README_REPRODUCE.md` and `REPO_MAP.md`; run its scripts from inside that folder). Frozen: nothing here is edited any more. |
| [`torch_actor/`](torch_actor/) | The working directory: the PyTorch port of the multi-head actor, the re-run of the draft's experiments, the checks and follow-up experiments made with it, and **the current benchmark, [`torch_actor/baseline_D4/`](torch_actor/baseline_D4/)**, on which new work is to be built. |

Start with [`torch_actor/README.md`](torch_actor/README.md).
