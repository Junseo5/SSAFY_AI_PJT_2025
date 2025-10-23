 # 251023_Kaggle_AllInOne — Report
 
 - Date: 2025-10-23
 - Score (Public LB): 0.80452
 - Notebook: `SSAFY_AI_PJT_2025/experiments/251023_Kaggle_AllInOne/Kaggle_AllInOne.ipynb`
 - Baseline: `SSAFY_AI_PJT_2025/experiments/251023_Baseline/251023_Baseline.ipynb` (0.76028)
 - Delta: +0.04424
 - Environment: Kaggle GPU (T4), Internet: On
 - Model: Qwen/Qwen2.5-VL-3B-Instruct
 - Image size: 384
 
 ## Changes vs Baseline
 - Single-file consolidation: install + env + inference in one notebook.
 - Kaggle-first installs: keep Kaggle Torch; `transformers` from Git; add `qwen-vl-utils[decord]==0.0.8`; minimize other packages.
 - Data path unification: `DATA_DIR` root with support for `train.csv`/`test.csv` either flat or under `data/`.
 - Default run is zero-shot inference (no training). Optional minimal training stub (disabled by default).
 - Outputs written to `/kaggle/working/submission_baseline.csv`.
 - Inference params: `max_new_tokens=10`, `do_sample=False`, `temperature=0.0`.
 
 ## Repro Steps (Kaggle)
 1) New Notebook → GPU: T4, Internet: On.
 2) Add data: dataset containing CSV + images.
 3) Set `DATA_DIR` to the mounted path (left Files panel).
 4) Run all cells → submission at `/kaggle/working/submission_baseline.csv`.
 
 ## Notes
 - Images must be present; otherwise a blank image fallback reduces accuracy.
 - First run downloads model/processor (longer setup time).
 
