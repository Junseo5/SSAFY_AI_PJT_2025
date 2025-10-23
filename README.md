# Kaggle VQA Challenge Solution

## 🎯 Project Overview

Visual Question Answering (VQA) 챌린지 솔루션으로, 이미지, 질문, 4개의 선택지를 입력받아 정답(a/b/c/d) 하나를 예측하는 시스템입니다.

- **Target Accuracy**: 85-88% (Top 10%)
- **Model**: Qwen2.5-VL-7B-Instruct (QLoRA 4-bit)
- **Hardware**: T4 GPU × 2 (30GB VRAM)
- **Strategy**: 3-fold Cross-Validation + Ensemble

## ⚠️ Critical T4 GPU Compatibility Notes

본 프로젝트는 T4 GPU 환경에서 실행되도록 최적화되었으며, 다음 사항이 적용되었습니다:

### 1. BFloat16 → Float16
- **문제**: T4는 BFloat16 미지원 (Ampere SM80+ 필요)
- **해결**: 모든 학습/추론에서 Float16 사용
- **영향**: `bf16=False`, `torch.float16` 사용

### 2. FlashAttention 2 제거
- **문제**: FA2는 Ampere 이상에서만 최적화
- **해결**: `attn_implementation="sdpa"` 사용 (기본 SDPA)
- **영향**: requirements.txt에서 flash-attn 제거

### 3. Transformers Git Install
- **문제**: PyPI 버전은 Qwen2.5-VL 지원 부족
- **해결**: Git에서 직접 설치
```bash
pip install git+https://github.com/huggingface/transformers.git
```

### 4. Qwen VL Utils 필수
```bash
pip install qwen-vl-utils[decord]==0.0.8
```

### 5. 라벨 정렬 교정 (가장 중요!)
- **문제**: 학습/추론 토큰 위치 불일치
- **해결**: Assistant 메시지에 정답 1글자 포함
```python
messages.append({
    "role": "assistant",
    "content": [{"type": "text", "text": answer}]  # 'a', 'b', 'c', 'd'
})
```

### 6. 프롬프트 템플릿 통일
- **방법**: `apply_chat_template` + `process_vision_info` 사용
- **장점**: 버전 호환성, 안정성

## 📁 Project Structure

```
SSAFY_AI_PJT_2025/
├── README.md
├── requirements.txt          # T4 호환 패키지 목록
├── install.sh               # 자동 설치 스크립트
├── config/
│   ├── train_config.yaml    # 학습 설정
│   ├── inference_config.yaml
│   ├── prompt_templates.yaml # 질문 유형별 프롬프트
│   └── normalize.yaml       # 정규화 규칙
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── images/              # 이미지 파일 (추가 필요)
├── scripts/
│   ├── __init__.py
│   ├── eda.py               # 탐색적 데이터 분석
│   ├── normalize.py         # 답변 정규화
│   ├── stratified_cv.py     # Stratified K-Fold
│   ├── augment.py           # 데이터 증강 (OCR 제외)
│   ├── prompt_manager.py    # 프롬프트 관리
│   ├── error_handler.py     # 에러 처리
│   ├── memory_optimizer.py  # GPU 메모리 관리
│   ├── baseline_train.py    # 🔵 Baseline 학습 (간단/빠름)
│   ├── baseline_infer.py    # 🔵 Baseline 추론
│   ├── train_lora.py        # 🟢 Advanced QLoRA 학습 (라벨 정렬 교정)
│   ├── infer_forced_choice.py # 🟢 Advanced Forced-choice 추론
│   ├── ensemble.py          # 앙상블 (확률 평균)
│   └── validate_submission.py # 제출 파일 검증
├── checkpoints/             # 모델 체크포인트
├── outputs/                 # 제출 파일
├── logs/                    # 학습 로그
└── notebooks/
    ├── 01_eda.ipynb
    └── 02_vqa_training.ipynb

```

## 🚀 Quick Start

### 1. Installation

```bash
# Option 1: 자동 설치
bash install.sh

# Option 2: 수동 설치
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install git+https://github.com/huggingface/transformers.git
pip install qwen-vl-utils[decord]==0.0.8
pip install -r requirements.txt
```

### 2. WandB Setup

```bash
wandb login
```

### 3. Data Preparation

```bash
# 이미지 파일을 data/images/ 폴더에 추가
# EDA 실행
python scripts/eda.py

# Stratified CV splits 생성
python scripts/stratified_cv.py
```

### 4. Training (Day 2)

```bash
# 7B 모델 3-fold 학습
for fold in 0 1 2; do
  python scripts/train_lora.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --fold $fold \
    --output_dir checkpoints/qwen-7b-fold$fold \
    --device cuda:0 \
    --num_epochs 3 \
    --lr 2e-5
done
```

### 5. Inference & Submission (Day 3)

```bash
# 각 fold별 추론
for fold in 0 1 2; do
  python scripts/infer_forced_choice.py \
    --model_path checkpoints/qwen-7b-fold$fold/final \
    --test_csv data/test.csv \
    --image_dir data/images \
    --output_csv outputs/submission_fold$fold.csv \
    --device cuda:0
done

# 앙상블
python scripts/ensemble.py \
  --predictions outputs/submission_fold0.csv outputs/submission_fold1.csv outputs/submission_fold2.csv \
  --weights 0.35 0.35 0.30 \
  --output outputs/submission_ensemble.csv

# 제출 파일 검증
python scripts/validate_submission.py --file outputs/submission_ensemble.csv
```

## 🎯 Two Workflows: Baseline vs Advanced

This project provides two workflows to suit different needs:

### 🔵 Baseline Workflow (Simple & Fast)

Perfect for quick testing and prototyping. Based on the competition's baseline notebook.

**Features**:
- Simple and straightforward
- Uses `AutoModelForVision2Seq` (compatible with baseline)
- Direct `path` column support (`train/train_0001.jpg`)
- No complex preprocessing
- Fast iteration

**Usage**:
```bash
# Training
python scripts/baseline_train.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --train_csv data/train.csv \
  --data_dir data \
  --output_dir checkpoints/baseline \
  --image_size 384 \
  --batch_size 1 \
  --epochs 1

# Inference
python scripts/baseline_infer.py \
  --model_path checkpoints/baseline \
  --test_csv data/test.csv \
  --data_dir data \
  --output_csv outputs/submission_baseline.csv

# Validation
python scripts/validate_submission.py --file outputs/submission_baseline.csv
```

**When to use**:
- Quick experimentation
- Baseline comparison
- Limited time/resources
- Testing new ideas quickly

### 🟢 Advanced Workflow (Optimized for Competition)

Full-featured workflow with all optimizations and critical fixes for maximum performance.

**Features**:
- `Qwen2_5_VLForConditionalGeneration` (latest class)
- Label alignment fix (assistant message)
- Question type-specific prompts (7 types)
- Stratified K-Fold Cross-Validation
- Data augmentation (with OCR protection)
- Ensemble methods (probability averaging)
- Answer normalization
- Memory optimization

**Usage**:
```bash
# 1. EDA & CV Splits
python scripts/eda.py
python scripts/stratified_cv.py

# 2. Training (3-fold)
for fold in 0 1 2; do
  python scripts/train_lora.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --fold $fold \
    --output_dir checkpoints/qwen-7b-fold$fold \
    --device cuda:0 \
    --num_epochs 3 \
    --lr 2e-5
done

# 3. Inference
for fold in 0 1 2; do
  python scripts/infer_forced_choice.py \
    --model_path checkpoints/qwen-7b-fold$fold/final \
    --test_csv data/test.csv \
    --image_dir data/images \
    --output_csv outputs/submission_fold$fold.csv \
    --device cuda:0
done

# 4. Ensemble
python scripts/ensemble.py \
  --predictions outputs/submission_fold*.csv \
  --method weighted \
  --val_accuracies 0.825 0.818 0.822 \
  --output outputs/submission_ensemble.csv

# 5. Validation
python scripts/validate_submission.py --file outputs/submission_ensemble.csv
```

**When to use**:
- Final competition submission
- Maximum accuracy required
- Multi-fold ensemble
- Full pipeline validation

### 📊 Expected Performance

| Workflow | Accuracy | Training Time | Notes |
|----------|----------|---------------|-------|
| Baseline (3B) | 75-78% | ~2h | Quick baseline |
| Advanced Single (7B) | 79-82% | ~4h/fold | QLoRA optimized |
| Advanced Ensemble (3-fold) | 83-85% | ~12h total | Full pipeline |
| Advanced + Optimization | 85-88% | ~15h total | HP tuning, TTA |

### 💡 Data Structure Support

Both workflows support flexible data structures:

```python
# Option 1: 'path' column (baseline style)
# data/train.csv:
# id,path,question,a,b,c,d,answer
# 1,train/train_0001.jpg,질문?,보기1,보기2,보기3,보기4,a

# Option 2: 'image' column (alternative)
# id,image,question,a,b,c,d,answer
# 1,images/train_0001.jpg,질문?,보기1,보기2,보기3,보기4,a
```

All scripts automatically detect and handle both formats.

## 📊 Key Features

### 1. 질문 유형별 최적화 프롬프트
- **counting**: 객체 카운팅 전문가 프롬프트
- **color**: 색상 인식 전문가 프롬프트
- **ocr**: OCR 전문가 프롬프트 (한글/영어/숫자)
- **yesno**: 시각적 추론 전문가 프롬프트
- **general**: 범용 VQA 프롬프트

### 2. 데이터 증강
- 보기 순서 셔플 + 정답 라벨 업데이트
- 한국어 질문 변형 (paraphrase)
- 이미지 증강 (밝기, 대비)
- **OCR 질문 제외**: 문자 반전 방지

### 3. Stratified K-Fold
- 질문 유형 비율 유지
- 정답 분포 균등화
- Seed 고정 (재현성)

### 4. QLoRA 학습
- 4-bit quantization (NF4)
- LoRA: r=24, alpha=48
- Language model만 학습 (Vision encoder 동결)
- Label smoothing: 0.05
- FP16 precision (T4 호환)

### 5. Forced-Choice 추론
- Logit-based 예측 (a/b/c/d 토큰 확률)
- Confidence 계산 (margin)
- 안전한 파싱 (fallback 포함)

### 6. 앙상블
- 3-fold 가중 투표
- 확률 평균 방식 (안정적)
- Validation 기반 가중치 조정

## 🔬 Architecture Details

### Model Configuration
```yaml
model: Qwen/Qwen2.5-VL-7B-Instruct
quantization: 4-bit NF4
precision: Float16 (T4 compatible)
attention: SDPA (FlashAttention 2 removed)
lora:
  r: 24
  alpha: 48
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Training Configuration
```yaml
batch_size: 4
gradient_accumulation_steps: 2
effective_batch_size: 8
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.05
num_epochs: 3
weight_decay: 0.01
label_smoothing: 0.05
optimizer: paged_adamw_8bit
fp16: true
bf16: false  # T4 unsupported
gradient_checkpointing: true
seed: 42
```

### Inference Configuration
```yaml
resolution:
  min_pixels: 256 * 28 * 28
  max_pixels: 768 * 28 * 28
  high_res: 1024 * 28 * 28  # 재추론용
generation:
  max_new_tokens: 1
  do_sample: false
  temperature: 0.0
```

## 📈 Expected Performance

| Stage | Accuracy | Notes |
|-------|----------|-------|
| Zero-shot Baseline | 65-68% | 프롬프트만 |
| Single Fold (7B) | 79-82% | QLoRA 학습 |
| 3-Fold Ensemble | 83-85% | 가중 평균 |
| + Optimization | 85-88% | HP tuning, TTA |

## 🔧 Troubleshooting

### 문제 1: ImportError: cannot import 'Qwen2_5_VLForConditionalGeneration'
```bash
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers.git
```

### 문제 2: ValueError: Bfloat16 is only supported...
- 모든 코드에서 `torch.bfloat16` → `torch.float16` 변경
- `bf16=False` 확인

### 문제 3: GPU Out of Memory
```python
# batch_size 줄이기
per_device_train_batch_size=2  # 4 → 2
gradient_accumulation_steps=4  # 2 → 4
```

### 문제 4: Validation accuracy가 25% 근처 (random guess)
- **원인**: 라벨 정렬 오류
- **해결**: `train_lora.py`에서 assistant 메시지 포함 확인

## 📝 Reproducibility

본 프로젝트는 완전한 재현성을 보장합니다:

- ✅ Random seed 고정: 42
- ✅ CUDNN deterministic: True
- ✅ Version-locked requirements.txt
- ✅ WandB experiment tracking
- ✅ Stratified CV with seed

## 🎓 Reference Documents

프로젝트 구현 시 참고한 문서:
- `FINAL_CORRECTED_Implementation_Prompt.md`: 최종 검증 버전 구현 가이드
- `VERIFICATION_SUMMARY.md`: 6가지 치명적 이슈 수정 사항
- `VQA_Specification_Enhancement.md`: 프롬프트 전략, 에러 처리 등

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Qwen Team: Qwen2.5-VL 모델
- Hugging Face: Transformers, PEFT
- WandB: Experiment tracking

---

**Generated for SSAFY AI Project 2025**

**Last Updated**: 2025-10-23

**Contact**: GitHub Issues
