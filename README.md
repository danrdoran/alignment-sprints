# alignment-sprints

## Sprint: Unsupervised Elicitation

## Overview
This project studies coherence from *Unsupervised Elicitation of Language Models*, which introduces Internal Coherence Maximization (ICM).

ICM searches for coherent labels on unlabeled examples, then uses those labels for training (in-context learning or supervised fine-tuning). The process uses minimal human input and can perform similarly to large amounts of gold supervision.

Potential applications include:
- Elicitation of latent knowledge and beliefs
- Improving honesty (consistency between actions and beliefs)
- Outer alignment (creating value specs / constitutions)

This sprint has two tasks:
- Reimplement ICM
- Critically analyze key methodological questions

## Part 1: Replication
Your first task is to implement ICM and reproduce TruthfulQA results. Recommended: spend about 1 hour reading the paper first.

### Task Requirements
- Implement ICM (Algorithm 1) **without** logical consistency fix
- Use the provided TruthfulQA mini dataset (256 train, 100 test)
- Run ICM to search for labels, then use those labels for **in-context learning** (not SFT)
- Generate a figure matching the TruthfulQA subfigure format from Figure 1, with the same 4 conditions
- Figure 1 in the paper is SFT; for this sprint, do in-context learning only
- Your figure should match format/trend (not exact visual details)
- Exact numbers may differ due to setup changes (e.g., smaller data), but trend should generally hold
- Use Python
- Review both the original repo and a popular reimplementation (code reuse is allowed)

### Model Choice
- Base: `Llama-3.1-405B`
- Chat: `Llama-3.1-405B-instruct`
- Use both through Hyperbolic API

Notes:
- You are expected to read API docs independently
- Set up your own API key
- Expected experiment cost is about `$20`

### Deliverables
- Submit code via the submission form (GitHub link or Google Drive zip)
- Include the main results figure in your repo

### Evaluation Criteria
- Accuracy of ICM reimplementation
- ICM few-shot accuracy from your implementation (bar chart)
- Code clarity (production-level polish is not required)

AI tools are allowed. If used, disclose:
- Which tools you used
- How you used them
- Links to relevant chat logs

Expected time allocation: **3-6 hours**

## Part 2: Critique
This part simulates research reasoning under uncertainty.

### Procedure
- Identify one important methodological weakness / limitation / questionable design choice
- Brainstorm practical fixes or simplifications
- Show reasoning transparently (uncertainties, assumptions, information sources)

### Deliverable
Submit a 1-2 page PDF report with:
- Your main critique (optionally mention alternatives considered) and why it matters
- How you would address it
- First test to reduce uncertainty
- Follow-up plan with more time

### Evaluation Criteria
Focus on conceptual and technical reasoning (not code-level details).

AI tools are allowed here too. If used, disclose tools/method and include relevant chat logs.

### Guidance: High-Value Critiques
Prefer issues that could materially change interpretation of a core paper claim.

Strong critique categories:
- Validity threats: does the measure capture the claimed construct?
- Completeness gaps: are key controls/baselines missing?
- Generalization concerns: does it transfer beyond the tested setup?
- Methodological robustness: are findings stable and reliable?

Expected time allocation: **1-2 hours**

## Local Project Structure
- `src/icm_ue/`: ICM implementation code (search, prompting, evaluation)
- `scripts/`: runnable entry points (`run_icm.py`, `run_eval.py`, `make_plot.py`)
- `configs/`: configuration and templates
- `outputs/`: generated artifacts (`logs/`, `labels/`, `metrics/`, `raw/`)
- `plots/`: final figures for submission
- `reports/`: critique write-up and final PDF
- `references/`: original and reimplementation reference repos
- `data/truthfulqa-mini/`: provided dataset

## Reproducibility
- Dependency lock file: `requirements.lock.txt`
- Run metadata template: `configs/RUN_MANIFEST_TEMPLATE.md`
- Record seed, hyperparameters, model IDs, and API settings for every run
