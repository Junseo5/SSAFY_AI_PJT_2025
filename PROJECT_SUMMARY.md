# Kaggle VQA Challenge - 프로젝트 요약

## 🎯 프로젝트 개요

Visual Question Answering (VQA) 챌린지를 위한 완전한 구현 프로젝트입니다.

- **모델**: Qwen2.5-VL-7B-Instruct (QLoRA 4-bit)
- **목표 정확도**: 85-88% (Top 10%)
- **환경**: T4 GPU × 2 (30GB VRAM)
- **기간**: 5일 해커톤

### 🎯 Two Workflows Available

- **🔵 Baseline Workflow**: 간단하고 빠른 프로토타이핑 (baseline_train.py, baseline_infer.py)
- **🟢 Advanced Workflow**: 최적화된 경쟁용 파이프라인 (train_lora.py, infer_forced_choice.py + ensemble)

### 💡 Data Structure Compatibility

모든 스크립트는 두 가지 데이터 형식을 자동으로 지원합니다:

- **Option 1**: `path` column (baseline style) - `train/train_0001.jpg`
- **Option 2**: `image` column (alternative) - `images/train_0001.jpg`

스크립트가 자동으로 감지하고 처리하므로 데이터 형식을 변경할 필요가 없습니다.

## ✅ 완료된 주요 작업

### 1. 프로젝트 구조 설정
- ✅ 디렉토리 구조 생성
- ✅ T4 호환 requirements.txt (FP16, FlashAttention 제거)
- ✅ 자동 설치 스크립트 (install.sh)
- ✅ 종합 README.md

### 2. 데이터 분석 및 전처리
- ✅ **EDA 스크립트** (eda.py)
  - 질문 유형 자동 분류 (counting, color, ocr, yesno, location, attribute, general)
  - 답변 형식 분석
  - 데이터 품질 체크
  - 시각화 생성

- ✅ **정규화 스크립트** (normalize.py)
  - 한글/영어 숫자 변환
  - 단위 정규화
  - 공백 및 특수문자 처리
  - Unicode 정규화 (NFKC)

- ✅ **Stratified CV** (stratified_cv.py)
  - 질문 유형 비율 유지
  - 정답 분포 균등화
  - Seed 고정 (재현성)

### 3. 데이터 증강
- ✅ **증강 스크립트** (augment.py)
  - 보기 순서 셔플 + 정답 라벨 자동 업데이트
  - 한국어 질문 변형 (paraphrase)
  - 이미지 증강 (밝기, 대비)
  - **✅ CRITICAL**: OCR 문제 이미지 증강 제외 (문자 반전 방지)

### 4. 프롬프트 엔지니어링
- ✅ **프롬프트 템플릿** (prompt_templates.yaml)
  - 질문 유형별 최적화 프롬프트 7종
  - Qwen2.5-VL 호환 형식

- ✅ **프롬프트 매니저** (prompt_manager.py)
  - `apply_chat_template` + `process_vision_info` 사용
  - 학습/추론용 메시지 생성
  - Assistant 메시지 자동 추가 (라벨 정렬)

### 5. 모델 학습 (가장 중요!)
- ✅ **학습 스크립트** (train_lora.py)
  - **✅ CRITICAL FIXES**:
    1. `Qwen2_5_VLForConditionalGeneration` (올바른 클래스명)
    2. `AutoProcessor` (올바른 프로세서)
    3. `torch.float16` (T4 호환, BF16 미사용)
    4. `attn_implementation="sdpa"` (FlashAttention 제거)
    5. `qwen_vl_utils.process_vision_info` (필수 사용)
    6. Assistant 메시지에 정답 포함 (라벨 정렬 교정)
    7. `add_generation_prompt=False` (학습 시)
  - QLoRA 4-bit (NF4)
  - LoRA: r=24, alpha=48
  - Language model만 학습 (Vision encoder 동결)
  - Label smoothing: 0.05
  - Gradient checkpointing
  - Seed 고정 (42)

### 6. 추론 및 예측
- ✅ **추론 스크립트** (infer_forced_choice.py)
  - Forced-choice 예측 (a/b/c/d)
  - Logit 기반 확률 계산
  - Confidence 측정 (margin)
  - 질문 유형별 프롬프트 자동 적용

### 7. 앙상블
- ✅ **앙상블 스크립트** (ensemble.py)
  - **✅ CRITICAL**: 확률 평균 방식 (안정적)
  - 가중 투표 (Validation 정확도 기반)
  - 3-fold 결과 통합

### 8. 검증 및 유틸리티
- ✅ **제출 파일 검증** (validate_submission.py)
  - 8단계 엄격한 검증
  - 일반적인 오류 자동 수정
  - 답변 분포 분석

- ✅ **에러 핸들러** (error_handler.py)
  - GPU OOM 자동 복구
  - 한글 토큰화 오류 방지
  - 모델 로딩 실패 처리
  - 안전한 추론 (재시도)

- ✅ **GPU 메모리 최적화** (memory_optimizer.py)
  - 메모리 기반 자동 설정
  - 메모리 모니터링
  - T4 최적화 설정

### 9. Jupyter Notebook
- ✅ **통합 노트북** (VQA_Training_Complete.ipynb)
  - 전체 파이프라인 통합
  - 단계별 실행 가능
  - 결과 분석 및 시각화

## 🔧 치명적 이슈 수정 (6가지)

### 1. Transformers 버전 & 클래스명
- ❌ 잘못: `Qwen2VLForConditionalGeneration`
- ✅ 올바름: `Qwen2_5_VLForConditionalGeneration`
- ✅ 필수: `transformers>=4.49.0` (Git install 권장)

### 2. T4 GPU BFloat16 미지원
- ❌ 잘못: `torch.bfloat16`, `bf16=True`
- ✅ 올바름: `torch.float16`, `fp16=True`, `bf16=False`

### 3. FlashAttention 2 미지원
- ❌ 잘못: `flash-attn==2.6.3`, `attn_implementation="flash_attention_2"`
- ✅ 올바름: FlashAttention 제거, `attn_implementation="sdpa"`

### 4. 라벨 정렬 오류 (가장 중요!)
- ❌ 잘못: 입력 마지막 토큰에 라벨 설정
- ✅ 올바름: Assistant 메시지에 정답 1글자 포함, 해당 토큰만 학습
- **예상 성능 향상**: +10-15pt

### 5. 수동 특수토큰 구성 금지
- ❌ 잘못: `<|vision_start|>` 등 문자열 직접 조립
- ✅ 올바름: `apply_chat_template` + `process_vision_info` 사용

### 6. 해상도 관리 통일
- ✅ `min_pixels/max_pixels` 파라미터로 일관 관리

## 📊 예상 성능

| 단계 | 정확도 | 비고 |
|------|--------|------|
| Zero-shot | 65-68% | 프롬프트만 |
| Single Fold (7B) | 79-82% | QLoRA 학습 |
| 3-Fold Ensemble | 83-85% | 가중 평균 |
| + Optimization | 85-88% | HP tuning, TTA |

## 🚀 실행 방법

### 🎯 Two Workflows Available

This project now supports two workflows:

#### 🔵 **Baseline Workflow** (간단/빠름)
Based on the competition's baseline notebook. Perfect for quick testing.

```bash
# 1. 학습
python scripts/baseline_train.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --train_csv data/train.csv \
  --data_dir data \
  --output_dir checkpoints/baseline \
  --epochs 1

# 2. 추론
python scripts/baseline_infer.py \
  --model_path checkpoints/baseline \
  --test_csv data/test.csv \
  --data_dir data \
  --output_csv outputs/submission_baseline.csv

# 3. 검증
python scripts/validate_submission.py --file outputs/submission_baseline.csv
```

#### 🟢 **Advanced Workflow** (최적화/고성능)
Full-featured with all optimizations for maximum competition performance.

### 1. 설치
```bash
bash install.sh
```

### 2. EDA 및 CV Splits
```bash
python scripts/eda.py
python scripts/stratified_cv.py
```

### 3. 학습 (3-fold)
```bash
for fold in 0 1 2; do
  python scripts/train_lora.py \
    --fold $fold \
    --output_dir checkpoints/qwen-7b-fold$fold \
    --device cuda:0
done
```

### 4. 추론
```bash
for fold in 0 1 2; do
  python scripts/infer_forced_choice.py \
    --model_path checkpoints/qwen-7b-fold$fold/final \
    --output_csv outputs/submission_fold$fold.csv
done
```

### 5. 앙상블
```bash
python scripts/ensemble.py \
  --predictions outputs/submission_fold*.csv \
  --method weighted \
  --val_accuracies 0.825 0.818 0.822 \
  --output outputs/submission_ensemble.csv
```

### 6. 검증
```bash
python scripts/validate_submission.py --file outputs/submission_ensemble.csv
```

## 📁 최종 파일 구조

```
SSAFY_AI_PJT_2025/
├── README.md                    # 프로젝트 설명서
├── PROJECT_SUMMARY.md           # 본 문서
├── requirements.txt             # T4 호환 패키지 목록
├── install.sh                   # 자동 설치 스크립트
├── config/
│   ├── prompt_templates.yaml    # 질문 유형별 프롬프트
│   └── normalize.yaml           # 정규화 규칙
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_with_folds.csv     # CV splits
│   └── images/
├── scripts/
│   ├── __init__.py
│   ├── eda.py                   # EDA 스크립트
│   ├── normalize.py             # 정규화
│   ├── stratified_cv.py         # Stratified CV
│   ├── augment.py               # 데이터 증강
│   ├── prompt_manager.py        # 프롬프트 관리
│   ├── error_handler.py         # 에러 처리
│   ├── memory_optimizer.py      # GPU 메모리 관리
│   ├── baseline_train.py        # 🔵 Baseline 학습 (간단/빠름)
│   ├── baseline_infer.py        # 🔵 Baseline 추론
│   ├── train_lora.py            # 🟢 ⭐ Advanced 학습 (라벨 정렬 교정)
│   ├── infer_forced_choice.py   # 🟢 Advanced 추론
│   ├── ensemble.py              # 앙상블 (확률 평균)
│   └── validate_submission.py   # 제출 파일 검증
├── notebooks/
│   └── VQA_Training_Complete.ipynb  # 통합 노트북
├── checkpoints/                 # 모델 체크포인트
├── outputs/                     # 제출 파일
└── logs/                        # 학습 로그
```

## 🎓 참고 문서

1. **FINAL_CORRECTED_Implementation_Prompt.md**: 최종 검증 버전 구현 가이드
2. **VERIFICATION_SUMMARY.md**: 6가지 치명적 이슈 수정 사항
3. **VQA_Specification_Enhancement.md**: 프롬프트 전략, 에러 처리

## ✅ 검증 체크리스트

- [x] ✅ Transformers Git install
- [x] ✅ Qwen2_5_VL* 클래스 사용
- [x] ✅ torch.float16 (T4 호환)
- [x] ✅ FlashAttention 제거
- [x] ✅ 라벨 정렬 교정 (assistant 메시지)
- [x] ✅ apply_chat_template + process_vision_info
- [x] ✅ Seed 고정 (42)
- [x] ✅ CUDNN deterministic
- [x] ✅ OCR 이미지 증강 제외

## 📝 다음 단계

### Day 4: 최적화
1. Hyperparameter optimization (Optuna)
2. High-resolution inference (1024px)
3. Test-time augmentation (TTA)
4. Ensemble weight tuning

### Day 5: 최종 제출
1. Error analysis
2. 타겟 증강
3. 다수결 재추론
4. 최종 제출 (4-5회)

## 🎯 목표 달성 전략

1. **Day 1-2**: 기반 구축 및 학습 (79-82%)
2. **Day 3**: 앙상블 (83-85%)
3. **Day 4**: 최적화 (85-87%)
4. **Day 5**: 최종 조정 (87-88%)

---

**프로젝트 완성도**: ✅ 95%

**남은 작업**: Hyperparameter tuning, Error analysis

**Generated for SSAFY AI Project 2025**

**Last Updated**: 2025-10-23
