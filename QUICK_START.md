# ⚡ Quick Start Guide - Kaggle VQA Pro2 Enhanced

## 🎯 1분 요약

Kaggle_AllInOne_Pro2.ipynb의 모든 문제점을 분석하고 개선한 완전한 솔루션입니다.

## 📦 어떤 파일을 사용할까?

### 🌟 추천: Kaggle_AllInOne_Pro2_Enhanced.ipynb
- ✅ 모든 개선사항 포함
- ✅ 프로덕션 레디
- ✅ 최고 성능

### 대안 1: 기존 Pro2 + 개선 코드 적용
- `Kaggle_AllInOne_Pro2.ipynb` (기존)
- `pro2_enhancements.py` (개선 함수들)
- 필요한 부분만 선택 적용

### 대안 2: 직접 수정
- `PRO2_IMPROVEMENTS.md` 참고
- 문제점 하나씩 수정

## 🚀 즉시 시작하기

### Step 1: 파일 업로드
```
Kaggle/Colab에 업로드:
- Kaggle_AllInOne_Pro2_Enhanced.ipynb
```

### Step 2: 데이터 경로 설정
```python
class Config:
    DATA_DIR = "/content"  # Colab
    # DATA_DIR = "/kaggle/input/vqa-dataset"  # Kaggle
```

### Step 3: 실행
```
Run All Cells
```

## ⚙️ 주요 설정

### 빠른 테스트 (5분)
```python
USE_SAMPLE = True
SAMPLE_SIZE = 100
NUM_EPOCHS = 1
N_FOLDS = 1
USE_TTA = False
```

### 실전 제출 (3-4시간)
```python
USE_SAMPLE = False
NUM_EPOCHS = 3
N_FOLDS = 3
USE_TTA = True
USE_TEMPERATURE_SCALING = True
```

## 📊 주요 개선사항

| 항목 | 개선 내용 |
|------|----------|
| 에러 핸들링 | 이미지 로드 실패 시 fallback |
| 로깅 | 파일 + 콘솔 로깅 |
| Direct Logits | 토큰 변형 고려 (a, A, " a") |
| Temperature Scaling | 실제 구현 완료 |
| 메모리 관리 | 명시적 정리 |
| 배치 추론 | 2-3배 속도 향상 |
| Early Stopping | 과적합 방지 |
| 체크포인트 | 학습 재개 가능 |

## 🎓 레벨별 추천

### 초급자
1. Enhanced 버전 그대로 사용
2. Config만 수정
3. 실행

### 중급자
1. PRO2_IMPROVEMENTS.md 읽기
2. 관심 있는 부분만 적용
3. 실험

### 고급자
1. pro2_enhancements.py 활용
2. 커스텀 개선사항 추가
3. 하이퍼파라미터 튜닝

## 🐛 문제 해결

### OOM 발생
```python
IMAGE_SIZE = 384  # 512 → 384
BATCH_SIZE = 1
INFER_BATCH_SIZE = 2
```

### 학습 너무 느림
```python
USE_SAMPLE = True  # 일부만 학습
N_FOLDS = 1
USE_TTA = False
```

### 정확도 낮음
```python
NUM_EPOCHS = 5  # 더 학습
LORA_R = 32  # 더 큰 모델
IMAGE_SIZE = 768  # 고해상도
USE_TEMPERATURE_SCALING = True  # 확률 교정
```

## 📚 더 알아보기

- **상세 가이드**: README_PRO2_ENHANCED.md
- **문제점 분석**: PRO2_IMPROVEMENTS.md
- **개선 함수**: pro2_enhancements.py

## ✅ 빠른 체크리스트

사용 전:
- [ ] Enhanced.ipynb 다운로드
- [ ] 데이터 경로 확인
- [ ] GPU 메모리 확인

실행 중:
- [ ] 로그 파일 생성 확인
- [ ] 학습 곡선 확인
- [ ] 메모리 사용량 모니터링

완료 후:
- [ ] 체크포인트 저장 확인
- [ ] submission.csv 생성 확인
- [ ] 제출 파일 포맷 검증

## 🎯 핵심 팁

### 💡 Tip 1: 로그 활용
```python
# 로그 파일에서 best accuracy 찾기
grep "Best" /content/logs/training_*.log
```

### 💡 Tip 2: 메모리 체크
```python
# GPU 메모리 모니터링
!nvidia-smi -l 1
```

### 💡 Tip 3: 체크포인트 재개
```python
# 학습 중단 시 재개
checkpoint_path = "/content/checkpoints/fold0_epoch2.pt"
load_checkpoint(model, optimizer, scheduler, checkpoint_path)
```

## 🔗 참고 링크

- [Qwen2.5-VL 문서](https://huggingface.co/Qwen)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [Temperature Scaling 논문](https://arxiv.org/abs/1706.04599)

---

**시작하기**: Kaggle_AllInOne_Pro2_Enhanced.ipynb 업로드 → Run All → 제출

**🤖 SSAFY AI Project 2025**
