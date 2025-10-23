# 251023_Kaggle_AllInOne_Pro — Report

* **Date:** 2025-10-24
* **Score (Public LB):** 0.82716
* **Notebook:** `SSAFY_AI_PJT_2025/experiments/251024_Kaggle_AllInOne_Pro/Kaggle_AllInOne_Pro.ipynb`
* **Baseline:** `SSAFY_AI_PJT_2025/experiments/251023_Kaggle_AllInOne/Kaggle_AllInOne.ipynb` (0.80452)
* **Delta:** +0.02264
* **Environment:** Kaggle GPU (Dual T4, multi-GPU enabled via accelerate)
* **Model:** Qwen/Qwen2.5-VL-3B-Instruct
* **Image size:** 384

---

## Changes vs Previous AllInOne

* **Multi-GPU runtime:** added `accelerate` configuration for distributed inference on dual T4 GPUs.
* **Version pinning:** fixed `transformers==4.45.2` for stability and compatibility.
* **API cleanup:** migrated to `AutoModelForImageTextToText` with `dtype=torch.float16`; removed deprecated arguments and warnings.
* **Prompt & parsing:** refined multiple-choice format and stricter single-letter answer extraction for consistent accuracy.
* **Inference:** deterministic decoding (`temperature=0.0`, `max_new_tokens=10`) with uniform `image_size=384`.
* **Workflow:** continued single-notebook (“all-in-one”) experiment pattern; every run archived under `/experiments/` with `REPORT.md` for score and change tracking.

---

## Repro Steps (Kaggle)

1. Create new notebook → GPU: Dual T4, Internet: On.
2. Add dataset (train/test CSV + images).
3. Set `DATA_DIR` to the mounted path (visible in left Files panel).
4. Run all cells → submission saved at `/kaggle/working/submission_pro.csv`.

---

## Notes

* Ensure images are present; fallback blanks reduce accuracy.
* First execution downloads model & processor (weights cached after first run).
* Deterministic setup verified across multiple Kaggle sessions (identical outputs within ±0.0001 variance).
