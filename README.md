# 📒 Kaggle VQA Challenge - 통합 노트북 버전

## 🎯 프로젝트 개요

Visual Question Answering (VQA) 챌린지를 위한 **단일 통합 노트북** 프로젝트입니다.

- **모델**: Qwen2.5-VL (3B/7B) + QLoRA
- **목표 정확도**: 85-88% (Top 10%)
- **환경**: T4 GPU 완벽 호환
- **특징**: 모든 기능이 하나의 노트북에 통합

## 🚀 빠른 시작

### 📒 메인 노트북

**`Kaggle_AllInOne_Pro.ipynb`** - 전체 파이프라인 통합 노트북

이 노트북 하나로 모든 것이 가능합니다:
- ✅ 환경 설정 및 패키지 설치
- ✅ Config 통합 관리
- ✅ 데이터 로드 및 EDA
- ✅ Stratified K-Fold CV
- ✅ 고급 학습 루프 (AMP, EMA, SWA, Cosine Warmup)
- ✅ TTA 추론
- ✅ 앙상블
- ✅ 제출 파일 생성

### 🔵 베이스라인 참고

**`251023_Baseline.ipynb`** - 경쟁 베이스라인 코드 (참고용)

## ✨ 주요 기능

### 1. T4 GPU 완벽 호환
- ✅ Float16 (BFloat16 대신)
- ✅ SDPA Attention (FlashAttention 제거)
- ✅ 4-bit QLoRA
- ✅ Gradient Checkpointing

### 2. 라벨 정렬 교정 (핵심!)
- ✅ Assistant 메시지에 정답 포함
- ✅ `add_generation_prompt=False` 사용
- ✅ 정답 토큰 위치 정확한 학습

### 3. 고급 학습 기법
- ✅ **AMP** (Automatic Mixed Precision)
- ✅ **EMA** (Exponential Moving Average)
- ✅ **SWA** (Stochastic Weight Averaging)
- ✅ **Cosine Warmup Scheduler**
- ✅ **Gradient Clipping**

### 4. K-Fold Cross-Validation
- ✅ Stratified K-Fold (답변 분포 유지)
- ✅ 3-Fold 기본 설정
- ✅ Fold별 독립 학습

### 5. TTA & Ensemble
- ✅ Test-Time Augmentation 지원
- ✅ Majority Voting 앙상블
- ✅ Weighted Ensemble 옵션

## 📊 예상 성능

| 설정 | 정확도 | 학습 시간 | 노트 |
|------|--------|-----------|------|
| Baseline (200 samples) | 60-65% | ~20min | 빠른 테스트 |
| Single Fold (3B, full data) | 75-78% | ~2h | 단일 모델 |
| 3-Fold Ensemble (3B) | 79-82% | ~6h | 앙상블 |
| 3-Fold Ensemble (7B) | 83-85% | ~12h | 고성능 |
| + TTA + Optimization (7B) | 85-88% | ~15h | 최고 성능 |

## 🗂️ 프로젝트 구조

```
SSAFY_AI_PJT_2025/
├── 📒 Kaggle_AllInOne_Pro.ipynb    ⭐ 메인 통합 노트북
├── 📒 251023_Baseline.ipynb         참고용 베이스라인
├── README.md                         이 파일
├── PROJECT_SUMMARY.md                프로젝트 요약
├── requirements.txt                  패키지 목록
├── install.sh                        자동 설치 스크립트
├── data/                             데이터 폴더
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── experiments/                      실험 결과 저장
│   └── README.md
├── checkpoints/                      모델 체크포인트 (학습 후 생성)
├── outputs/                          제출 파일 (추론 후 생성)
└── logs/                             학습 로그 (선택)
```

## 🎓 사용 방법

### 1. 환경 준비 (Colab/Kaggle)

```python
# Kaggle_AllInOne_Pro.ipynb의 첫 번째 코드 셀 실행
!pip install -q "transformers>=4.44.2" "accelerate>=0.34.2" "peft>=0.13.2" \
    "bitsandbytes>=0.43.1" datasets pillow pandas torch torchvision \
    scikit-learn matplotlib seaborn tqdm --upgrade
!pip install -q qwen-vl-utils==0.0.8

# 런타임 재시작
```

### 2. 데이터 업로드

Colab의 경우:
```python
from google.colab import drive
drive.mount('/content/drive')

# 데이터 압축 해제
!unzip "/content/drive/My Drive/data.zip" -d "/content/"
```

Kaggle의 경우:
- Add Data → Upload Dataset

### 3. Config 설정

노트북의 Config 셀에서 하이퍼파라미터 조정:

```python
class Config:
    # 모델 설정
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # 또는 7B
    IMAGE_SIZE = 384  # 384, 512, 768

    # K-Fold 설정
    N_FOLDS = 3
    USE_KFOLD = True

    # 학습 설정
    NUM_EPOCHS = 1  # 실전: 3~5
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 1e-4

    # 고급 기법
    USE_AMP = True
    USE_EMA = True
    USE_SWA = False
    USE_TTA = False

    # 샘플링 (디버깅)
    USE_SAMPLE = True  # False: 전체 데이터
    SAMPLE_SIZE = 200
```

### 4. 순차 실행

노트북의 모든 셀을 순서대로 실행:
1. 환경 설정
2. Config
3. 데이터 로드 & EDA
4. K-Fold 생성
5. Dataset 정의
6. 모델 로드
7. 학습
8. 추론
9. 앙상블
10. 결과 분석

### 5. 제출

`outputs/submission_ensemble.csv` (또는 `submission_single.csv`) 파일을 다운로드하여 제출

## 🔧 하이퍼파라미터 튜닝 가이드

### 빠른 테스트 (20분)
```python
USE_SAMPLE = True
SAMPLE_SIZE = 200
NUM_EPOCHS = 1
USE_KFOLD = False
```

### 단일 모델 실험 (2시간)
```python
USE_SAMPLE = False
NUM_EPOCHS = 3
USE_KFOLD = False
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
```

### 3-Fold 앙상블 (6-12시간)
```python
USE_SAMPLE = False
NUM_EPOCHS = 3
USE_KFOLD = True
N_FOLDS = 3
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # 고성능
```

### 최고 성능 (15시간)
```python
USE_SAMPLE = False
NUM_EPOCHS = 5
USE_KFOLD = True
N_FOLDS = 3
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_SIZE = 512  # 또는 768
USE_EMA = True
USE_SWA = True
USE_TTA = True
TTA_SCALES = [0.9, 1.0, 1.1]
```

## ⚠️ 중요 사항

### T4 GPU 호환성
- **Float16 사용** (BFloat16 아님) - T4는 BF16 미지원
- **SDPA Attention** (FlashAttention 제거) - T4 최적화 불가
- **4-bit Quantization** - 메모리 효율

### 라벨 정렬 교정
이것이 가장 중요한 수정 사항입니다!

❌ **잘못된 방법** (학습/추론 불일치):
```python
# 학습 시 정답 없이 학습
messages = [
    {"role": "user", "content": [...]},
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
```

✅ **올바른 방법** (라벨 정렬):
```python
# 학습 시 정답 포함
messages = [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": [{"type": "text", "text": "a"}]}  # 정답!
]
text = processor.apply_chat_template(messages, add_generation_prompt=False)  # False!
```

### 재현성
- Seed 42로 고정
- `torch.backends.cudnn.deterministic = True`

### 메모리 관리
- Gradient Checkpointing 활성화
- Batch Size 1 + Gradient Accumulation 4

## 📌 FAQ

### Q1: OOM (Out of Memory) 에러가 발생해요
**A**: 다음을 시도하세요:
- `BATCH_SIZE = 1`로 감소
- `IMAGE_SIZE = 384`로 감소
- `MODEL_ID`를 3B로 변경
- `USE_EMA = False`, `USE_SWA = False`

### Q2: 학습이 너무 느려요
**A**:
- `USE_SAMPLE = True`, `SAMPLE_SIZE = 200`으로 빠른 테스트
- `NUM_EPOCHS = 1`로 감소
- `USE_KFOLD = False`로 단일 모델 학습

### Q3: 정확도가 낮아요
**A**:
- `NUM_EPOCHS` 증가 (3~5)
- `MODEL_ID`를 7B로 변경
- `IMAGE_SIZE` 증가 (512, 768)
- `USE_KFOLD = True`로 앙상블
- `USE_EMA = True`, `USE_TTA = True`

### Q4: scripts/ 폴더가 없어요
**A**: 모든 코드가 `Kaggle_AllInOne_Pro.ipynb` 노트북에 통합되어 있습니다. 별도 스크립트 파일이 필요 없습니다.

## 📚 참고 자료

- **Qwen2.5-VL 공식 문서**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **PEFT (LoRA)**: https://huggingface.co/docs/peft
- **Transformers**: https://huggingface.co/docs/transformers

## 📊 변경 사항 (이전 버전 대비)

### ✅ 통합 완료
- ❌ `scripts/` 폴더 → ✅ 노트북에 통합
- ❌ `config/` 폴더 → ✅ Config 클래스로 통합
- ❌ `notebooks/VQA_Training_Complete.ipynb` → ✅ `Kaggle_AllInOne_Pro.ipynb`로 대체

### ✅ 추가된 기능
- ✅ EMA (Exponential Moving Average)
- ✅ SWA (Stochastic Weight Averaging)
- ✅ Cosine Warmup Scheduler
- ✅ TTA (Test-Time Augmentation)
- ✅ 통합 Config 관리
- ✅ 자동 EDA & 시각화

### ✅ 유지된 기능
- ✅ T4 호환성 (Float16, SDPA)
- ✅ 라벨 정렬 교정
- ✅ Stratified K-Fold
- ✅ QLoRA (4-bit)
- ✅ Gradient Checkpointing

## 🎯 다음 단계

1. **실험 관리**: `experiments/` 폴더에 실험 로그 저장
2. **하이퍼파라미터 최적화**: Optuna 등 활용
3. **앙상블 개선**: Weighted Voting, Stacking
4. **데이터 증강**: Choice Shuffle, Paraphrase
5. **에러 분석**: 예측 실패 샘플 분석

## 📧 문의

- **GitHub Issues**: 프로젝트 관련 질문

---

**🤖 SSAFY AI Project 2025**

**⭐ 행운을 빕니다!**
