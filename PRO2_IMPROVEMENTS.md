# 📋 Kaggle_AllInOne_Pro2 개선사항 및 수정 가이드

## 🔍 발견된 주요 문제점

### 1. **에러 핸들링 부족** ⚠️
**문제**: 이미지 로드 실패, 모델 로드 실패 시 프로그램이 중단됨
**영향도**: 🔴 High - 런타임 중 크래시 가능

**현재 코드 (문제):**
```python
img = Image.open(img_path).convert("RGB")  # 실패 시 즉시 중단
```

**개선 코드:**
```python
def _load_image_safe(img_path, fallback_size=512):
    """안전한 이미지 로드 with fallback"""
    try:
        img = Image.open(img_path).convert("RGB")
        if img.size[0] < 10 or img.size[1] < 10:
            raise ValueError(f"Image too small: {img.size}")
        return img, True
    except Exception as e:
        logger.warning(f"⚠️ Image load failed ({img_path}): {e}")
        return Image.new('RGB', (fallback_size, fallback_size), color='white'), False
```

### 2. **Direct Logits 로직 부정확** 🎯
**문제**: 단일 토큰만 고려, 멀티-토큰 답변(예: "a ", " a") 처리 미흡
**영향도**: 🟡 Medium - 추론 정확도 저하 가능

**현재 코드 (문제):**
```python
# 각 choice의 토큰 ID만 추출, 공백/변형 고려 안함
token_ids = processor.tokenizer.encode(choice, add_special_tokens=False)
```

**개선 코드:**
```python
def get_choice_token_ids_robust(processor):
    """강화된 토큰 ID 추출 - 변형 고려"""
    choice_tokens = {}
    for choice in ['a', 'b', 'c', 'd']:
        # 여러 변형 고려
        variants = [
            choice,           # "a"
            f" {choice}",     # " a"
            f"{choice} ",     # "a "
            choice.upper(),   # "A"
        ]
        all_token_ids = set()
        for variant in variants:
            token_ids = processor.tokenizer.encode(variant, add_special_tokens=False)
            all_token_ids.update(token_ids)
        choice_tokens[choice] = list(all_token_ids)
    return choice_tokens
```

### 3. **Temperature Scaling 미구현** 🌡️
**문제**: 코드만 있고 실제로 사용되지 않음
**영향도**: 🟡 Medium - 확률 교정 미적용

**개선 코드:**
```python
class TemperatureScaler:
    """Temperature Scaling for Calibration"""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        """검증 세트로 최적 temperature 학습"""
        from scipy.optimize import minimize

        def nll_loss(temp):
            temp = max(temp[0], 0.01)  # 최소값 방지
            scaled_probs = F.softmax(torch.tensor(logits) / temp, dim=1).numpy()
            # Negative log-likelihood
            nll = -np.log(scaled_probs[np.arange(len(labels)), labels] + 1e-10).mean()
            return nll

        result = minimize(nll_loss, x0=[1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        logger.info(f"✅ Optimal Temperature: {self.temperature:.4f}")
        return self

    def transform(self, logits):
        """학습된 temperature로 확률 스케일"""
        return F.softmax(torch.tensor(logits) / self.temperature, dim=1).numpy()

# 사용 예시:
# scaler = TemperatureScaler().fit(val_logits, val_labels)
# calibrated_probs = scaler.transform(test_logits)
```

### 4. **배치 추론 미구현** 🚀
**문제**: Config에만 있고 실제 구현 없음
**영향도**: 🟢 Low - 성능 개선 기회 손실

**개선 코드:**
```python
def infer_batch(model, processor, test_df, batch_size=4, tta_scales=[1.0]):
    \"\"\"배치 추론 구현\"\"\"
    model.eval()
    all_predictions = []
    all_probs = []

    # DataLoader 생성
    test_ds = VQADataset(test_df, processor, cfg.DATA_DIR, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(processor, train=False),
        num_workers=0
    )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=\"Batch Inference\"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # 각 샘플별 확률 계산
            for i in range(len(logits)):
                choice_probs = extract_choice_probs(logits[i], processor)
                pred = max(choice_probs, key=choice_probs.get)
                all_predictions.append(pred)
                all_probs.append(choice_probs)

    return all_predictions, all_probs
```

### 5. **로깅 시스템 부재** 📝
**문제**: print만 사용, 로그 파일 저장 안됨
**영향도**: 🟡 Medium - 디버깅 어려움

**개선 코드:**
```python
import logging
from datetime import datetime

def setup_logging(log_dir, level=logging.INFO):
    \"\"\"로깅 시스템 설정\"\"\"
    logger = logging.getLogger('VQA')
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 파일 핸들러
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f\"{log_dir}/training_{timestamp}.log\")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging(cfg.LOG_DIR)
logger.info(\"✅ Training started\")
```

### 6. **메모리 최적화 부족** 💾
**문제**: 명시적 메모리 정리 없음
**영향도**: 🟡 Medium - OOM 가능성

**개선 코드:**
```python
def clear_memory():
    \"\"\"메모리 정리\"\"\"
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Fold 간 사용
for fold in cfg.TRAIN_FOLDS:
    train_fold(...)
    clear_memory()
    logger.info(f\"💾 Memory cleared after fold {fold}\")
```

### 7. **체크포인트 재개 기능 없음** 💿
**문제**: 학습 중단 시 처음부터 다시 시작
**영향도**: 🟡 Medium - 시간 낭비

**개선 코드:**
```python
def save_checkpoint(model, optimizer, epoch, fold, metrics, path):
    \"\"\"체크포인트 저장\"\"\"
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    logger.info(f\"💾 Checkpoint saved: {path}\")

def load_checkpoint(model, optimizer, path):
    \"\"\"체크포인트 로드\"\"\"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info(f\"✅ Checkpoint loaded: epoch {checkpoint['epoch']}\")
    return checkpoint['epoch'], checkpoint['metrics']
```

### 8. **검증 데이터 처리 일관성 문제** ⚖️
**문제**: valid_loader도 train=True collator 사용
**영향도**: 🟢 Low - 이미 수정됨

**수정 확인:**
```python
# ✅ 이미 올바르게 수정됨
valid_ds = VQADataset(valid_subset, processor, cfg.DATA_DIR, train=False)
valid_loader = DataLoader(
    valid_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
    collate_fn=DataCollator(processor, train=False),  # ✅ train=False
    num_workers=0
)
```

## 🎯 우선순위별 적용 가이드

### 우선순위 1 (필수) - 즉시 적용
1. ✅ **에러 핸들링** - Dataset/DataCollator에 try-except 추가
2. ✅ **로깅 시스템** - logger 설정 및 사용
3. ✅ **메모리 정리** - Fold 간 clear_memory() 호출

### 우선순위 2 (권장) - 성능 개선
4. ✅ **Direct Logits 개선** - 토큰 변형 고려
5. ✅ **Temperature Scaling** - TemperatureScaler 클래스 추가
6. ✅ **배치 추론** - infer_batch 함수 구현

### 우선순위 3 (선택) - 편의성
7. ✅ **체크포인트** - save/load_checkpoint 추가
8. ✅ **Early Stopping** - patience 기반 조기 종료

## 📦 빠른 적용 가이드

### 방법 1: 기존 Pro2 파일 수정
기존 `Kaggle_AllInOne_Pro2.ipynb` 파일의 해당 셀을 위 개선 코드로 교체

### 방법 2: Enhanced 버전 사용
`Kaggle_AllInOne_Pro2_Enhanced.ipynb` 파일 사용 (모든 개선사항 포함)

### 방법 3: 점진적 적용
1. 먼저 에러 핸들링만 추가
2. 로깅 시스템 추가
3. 나머지 기능 순차 추가

## 🔧 주요 Config 변경 사항

```python
class Config:
    # ... 기존 설정 ...

    # 추가된 설정
    LOG_DIR = f\"{DATA_DIR}/logs\"  # 로그 디렉토리
    LOG_LEVEL = logging.INFO  # 로그 레벨
    SAVE_EVERY_EPOCH = True  # 에폭마다 체크포인트
    USE_EARLY_STOPPING = False  # Early stopping 사용 여부
    EARLY_STOPPING_PATIENCE = 2  # Early stopping patience
    USE_BATCH_INFERENCE = True  # 배치 추론 사용
    FOLD_WEIGHTS = None  # 앙상블 가중치 [0.4, 0.3, 0.3] 등
```

## 📊 예상 개선 효과

| 개선사항 | 정확도 향상 | 속도 향상 | 안정성 향상 |
|---------|------------|----------|-----------|
| 에러 핸들링 | - | - | ⭐⭐⭐ |
| Direct Logits 개선 | ⭐ | - | ⭐ |
| Temperature Scaling | ⭐⭐ | - | ⭐ |
| 배치 추론 | - | ⭐⭐ | - |
| 로깅 시스템 | - | - | ⭐⭐ |
| 메모리 최적화 | - | ⭐ | ⭐⭐ |

## 🚀 권장 실행 순서

1. **개선된 Config 설정** → 로깅 활성화
2. **Dataset/DataCollator 업데이트** → 에러 핸들링
3. **Training Loop 개선** → 체크포인트 + Early stopping
4. **Inference 개선** → Direct Logits 강화 + 배치 추론
5. **Temperature Scaling 적용** → 확률 교정
6. **Ensemble 개선** → 가중 앙상블

## 📝 테스트 체크리스트

- [ ] 이미지 로드 실패 시 fallback 동작 확인
- [ ] 로그 파일 생성 확인
- [ ] 체크포인트 저장/로드 확인
- [ ] Temperature scaling 적용 확인
- [ ] 배치 추론 정상 동작 확인
- [ ] 메모리 정리 확인
- [ ] 최종 제출 파일 포맷 확인

---

**🤖 SSAFY AI Project 2025 - Pro2 Improvements**
**📅 Last Updated: 2025-10-24**
