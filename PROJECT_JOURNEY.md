# 🚀 Visual Question Answering 프로젝트 성장 과정

## 📌 프로젝트 개요
**목표**: Kaggle VQA 챌린지에서 Top 10% 달성 (목표 정확도 85-88%)
**기간**: 2025-10-23 ~ 2025-10-27
**최종 성과**: 0.76028 → 0.92386 (+21.5% 향상) 🏆

---

## 🎯 Phase 1: 기반 구축 및 아키텍처 설계 (2025-10-23)

### 1단계: 프로젝트 초기화 및 데이터 준비
**커밋**: `8ba0cf5` Initial commit → `cf428aa` Base data

#### 주요 작업
- 프로젝트 구조 설정
- 베이스 데이터셋 준비
- 프롬프트 마크다운 파일 저장

---

### 2단계: T4 GPU 호환성 확보 및 핵심 아키텍처 구현
**커밋**: `ed8413b` Complete VQA project implementation with critical fixes

#### 🔥 핵심 트러블슈팅

**Problem 1: T4 GPU BFloat16 미지원**
```
❌ 문제: T4 GPU는 BFloat16을 네이티브로 지원하지 않아 성능 저하
✅ 해결: torch.float16으로 변경, SDPA Attention 사용
```

**Problem 2: FlashAttention 호환성 이슈**
```
❌ 문제: FlashAttention 2가 T4에서 최적화 불가
✅ 해결: attn_implementation="sdpa"로 변경
```

**Problem 3: 라벨 정렬 불일치 (가장 중요!)**
```
❌ 문제: 학습 시 정답 토큰 위치와 추론 시 생성 위치 불일치
✅ 해결: Assistant 메시지에 정답 포함 + add_generation_prompt=False
```
```python
# Before (잘못된 방법)
messages = [{"role": "user", "content": [...]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)

# After (올바른 방법)
messages = [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": [{"type": "text", "text": "a"}]}
]
text = processor.apply_chat_template(messages, add_generation_prompt=False)
```

**Problem 4: 모델 클래스 선택**
```
❌ 문제: AutoModelForVision2Seq 사용 시 성능 저하
✅ 해결: Qwen2_5_VLForConditionalGeneration 직접 사용
```

#### 성과
- **완전한 파이프라인 구축**: 16개 Python 스크립트 + 완전 통합 노트북
- **QLoRA 최적화**: 4-bit quantization, r=24, alpha=48
- **프롬프트 엔지니어링**: 7가지 질문 유형별 템플릿
- **Stratified K-Fold CV**: 3-fold 앙상블 시스템
- **예상 성능**: 65-68% (zero-shot) → 83-85% (3-fold ensemble)

---

### 3단계: 데이터 호환성 확보
**커밋**: `051a5ef` Add baseline workflow and improve data structure compatibility

#### 트러블슈팅
**Problem: 데이터 컬럼명 불일치**
```
❌ 문제: 베이스라인은 'path' 컬럼, 고급 버전은 'image' 컬럼 사용
✅ 해결: 자동 감지 로직 추가 (path/image 모두 지원)
```

#### 성과
- **Dual Workflow**: Baseline(빠름) + Advanced(최적화) 동시 지원
- **완벽한 데이터 호환성**: 'path'/'image' 컬럼 자동 감지

---

## 🎯 Phase 2: 통합 노트북 전환 및 성능 최적화 (2025-10-24)

### 4단계: 통합 노트북 아키텍처 구축
**커밋**: `3e41494` Consolidate project into single all-in-one notebook

#### 전략적 전환
```
AS-IS: 분산된 스크립트 (scripts/, config/, notebooks/)
TO-BE: 단일 통합 노트북 (Kaggle_AllInOne_Pro.ipynb)
```

#### 고급 기법 추가
- **EMA (Exponential Moving Average)**: 모델 가중치 안정화
- **SWA (Stochastic Weight Averaging)**: 일반화 성능 향상
- **Cosine Warmup Scheduler**: 안정적 학습률 스케줄링
- **TTA (Test-Time Augmentation)**: 추론 시 앙상블

#### 성과
- **개발 효율성**: 모든 기능이 하나의 노트북에 통합
- **실험 관리**: experiments/ 폴더로 버전별 관리 시작

---

### 5단계: 첫 실전 제출 및 성능 검증
**커밋**: `7e898c4` Add Kaggle_AllInOne (0.80452)

#### Public Leaderboard 첫 점수
```
📊 Score: 0.80452
📈 Baseline 대비: +4.424% 향상
```

#### 성공 요인
- **단일 노트북 워크플로우**: 실험별 버전 관리
- **REPORT.md**: 각 실험의 점수와 변경사항 추적
- **일관된 환경**: transformers==4.45.2 고정

---

### 6단계: Multi-GPU 최적화 및 점수 향상
**커밋**: `7627f43` Kaggle_AllInOne_Pro reaches 0.82716

#### 🔥 핵심 최적화

**Optimization 1: Multi-GPU 활용**
```
✅ Dual T4 GPU 병렬 처리
✅ accelerate 라이브러리 통합
✅ 추론 속도 2배 향상
```

**Optimization 2: API 현대화**
```
Before: AutoModelForVision2Seq (deprecated warnings)
After: AutoModelForImageTextToText + dtype=torch.float16
```

**Optimization 3: 프롬프트 정교화**
```
✅ 다중 선택 질문 템플릿 강화
✅ 단일 문자 답변 파서 엄격화
✅ 일관된 이미지 크기 (384px)
```

**Optimization 4: 결정론적 추론**
```python
generation_config = {
    "temperature": 0.0,  # 결정론적
    "max_new_tokens": 10,  # 답변만
    "do_sample": False
}
```

#### 성과
```
📊 Score: 0.82716
📈 Delta: +0.02264 (+2.8% 향상)
📈 총 향상: +8.8% (베이스라인 대비)
```

---

### 7단계: 라벨 마스킹 및 Direct Logits 도입
**커밋**: `a71304f` Pro2 정확도 향상

#### 🔥 고급 트러블슈팅

**Problem 1: 프롬프트 토큰 학습 비효율**
```
❌ 문제: 전체 시퀀스 손실 계산 → 프롬프트 토큰도 학습 대상
✅ 해결: 라벨 마스킹 (answer 토큰만 감독)
```
```python
# Assistant 정답 토큰만 학습
labels = [-100] * len(prompt_tokens) + answer_token_ids
```

**Problem 2: 생성 기반 추론 불안정성**
```
❌ 문제: generate() 사용 시 비결정론적 출력
✅ 해결: Direct Logits 추론
```
```python
# a, b, c, d 토큰의 logit 값 직접 계산
logits = model(**inputs).logits
probs = F.softmax(logits[answer_position, token_ids], dim=-1)
```

**Problem 3: 검증 데이터 정답 주입**
```
❌ 문제: valid_ds에 train=True → 정답 텍스트 포함되어 성능 과대평가
✅ 해결: valid_ds에 train=False 플래그
```

**Problem 4: TTA 구현**
```
✅ 해결: [0.9, 1.0, 1.1] 스케일 평균으로 robustness 향상
```

#### 성과
- **학습 효율성**: Answer 토큰만 감독 → 수렴 속도 향상
- **추론 안정성**: Direct Logits → 일관된 예측
- **앙상블 강화**: 확률 평균 → argmax (Majority Voting 대체)

---

### 8단계: Pro2 Enhanced - 8가지 문제점 해결
**커밋**: `f7a162d` Add comprehensive Pro2 enhancements

#### 🔥 Production-Ready 개선

**1. 에러 핸들링 강화**
```python
# 이미지 로드 실패 시 fallback
try:
    image = Image.open(path)
except:
    image = Image.new('RGB', (384, 384))  # 빈 이미지
```

**2. 로깅 시스템**
```python
logging.basicConfig(
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

**3. Direct Logits 정교화**
```python
# 토큰 변형 고려: a, A, " a"
token_variations = [
    tokenizer.encode("a")[0],
    tokenizer.encode("A")[0],
    tokenizer.encode(" a")[0]
]
prob_a = sum(probs[token_variations])
```

**4. Temperature Scaling**
```python
# 검증 세트로 확률 교정
temperature = find_optimal_temperature(val_logits, val_labels)
calibrated_probs = F.softmax(logits / temperature, dim=-1)
```

**5. 배치 추론**
```
✅ 속도: 1x → 2-3x 향상
```

**6. Early Stopping**
```python
if val_acc < best_val_acc - patience:
    break  # 과적합 방지
```

**7. 체크포인트 관리**
```python
# 학습 재개 가능
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f'checkpoint_epoch_{epoch}.pt')
```

**8. 메모리 최적화**
```python
torch.cuda.empty_cache()
gc.collect()
```

#### 성과
- **정확도**: 85-87% → 87-89% (+2% 예상)
- **추론 속도**: 2-3배 향상
- **안정성**: Production-ready 수준
- **재사용성**: pro2_enhancements.py 모듈화

---

## 🎯 Phase 3: 대규모 모델 실험 및 한계 극복 (2025-10-24)

### 9단계: Qwen3-VL-30B Multi-GPU 시도
**커밋**: `8eb669a` Add Qwen3-VL-30B Multi-GPU support

#### 야심찬 시도
```
목표: 30B 파라미터 모델로 88-90% 정확도 달성
환경: T4 * 2 (총 32GB)
전략: Model Parallelism + 4-bit Quantization
```

#### 🔥 대규모 모델 최적화 트러블슈팅

**Problem 1: OOM (Out of Memory)**
```
❌ 문제: 30B 모델은 T4*2로도 메모리 부족
✅ 해결 시도:
  - 4-bit Quantization (75% 메모리 절감)
  - Gradient Checkpointing (40% 활성화 메모리 절감)
  - CPU Offloading (Optimizer states)
  - Double Quantization
```

**Problem 2: Model Parallelism 구현**
```python
# device_map="auto"로 자동 분산
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    max_memory={0: "14GB", 1: "14GB"},
    quantization_config=bnb_config
)
```

**Problem 3: 학습 설정 극한 최적화**
```python
BATCH_SIZE = 1            # 필수
GRAD_ACCUM_STEPS = 16     # 효과적 배치 크기 16
LORA_R = 8                # 작은 rank (30B에 맞춤)
IMAGE_SIZE = 384          # 512는 OOM
```

#### 성과 (이론적)
- **예상 정확도**: 88-90% (+3~5% vs 3B)
- **메모리 사용**: GPU0 13GB, GPU1 13GB
- **학습 속도**: ~2min/epoch (3B 대비 2배 느림)

#### 핵심 기여
- **완전한 문서화**: QWEN3_30B_GUIDE.md, QUICK_START.md
- **재사용 가능 코드**: qwen3_30b_multigpu_core.py
- **실행 가능 노트북**: Kaggle_Qwen3_30B_MultiGPU.ipynb

---

### 10단계: FP8 Pre-Quantization 실험
**커밋**: `e2a12ba` Upgrade to Qwen3-VL-30B-A3B-Instruct-FP8

#### 최첨단 기술 시도
```
전략: FP8 Pre-quantization + 4-bit Re-quantization
목표: BF16 수준 정확도 + 메모리 효율
```

#### 🔥 FP8 최적화 트러블슈팅

**Problem 1: T4 FP8 미지원**
```
✅ 해결: FP8 → FP16 자동 변환 fallback
```

**Problem 2: Dual Quantization 안정성**
```
전략: Fine-grained FP8 (block size 128) + BitsAndBytes 4-bit
```

**Problem 3: 추론 시 모델 로딩 실패**
```python
# Fallback 메커니즘
try:
    model = load_finetuned_fp8_model()
except:
    model = load_base_fp8_model()  # 베이스 모델로 fallback
```

#### 성과 (이론적)
- **메모리**: 12-14GB per GPU (vs 13-15GB before)
- **정확도**: 88-91% (BF16 동등)
- **안정성**: 3단계 에러 핸들링

---

### 11단계: 30B 모델 폐기 결정
**커밋**: `e91dc9f` 폴더 정리 및 모델 업그레이드

#### 🔥 현실적 판단

**결정**: 30B 모델 완전 폐기
```
❌ 문제점:
  - T4 GPU로는 실질적 학습 불가능
  - 메모리 제약으로 IMAGE_SIZE 384 고정
  - 학습 속도 너무 느림 (~2min/epoch)
  - OOM 위험 상존

✅ 대안: Qwen3-VL-8B-Instruct로 전환
  - 3B → 8B: 적절한 성능 향상
  - T4에서 안정적 학습 가능
  - IMAGE_SIZE 768까지 가능
```

#### 교훈
> "더 큰 모델이 항상 답은 아니다. 주어진 하드웨어에서 최적화된 중간 모델이 더 현실적이다."

---

## 🎯 Phase 4: 최적화 완결 및 최종 튜닝 (2025-10-25 ~ 2025-10-27)

### 12단계: Qwen3-VL-8B 전환 및 로직 개선
**커밋**: `4209cf5` 최적화 진행 폴더 정리 및 성능 최대 향상

#### 전략적 재정비
```
모델: Qwen2.5-VL-3B → Qwen3-VL-8B-Instruct
기대: 단순 모델 업그레이드만으로도 큰 성능 향상
```

#### 🔥 학습 로직 개선
```
❌ 기존: 비효율적 학습 루프
✅ 개선:
  - 메모리 관리 최적화
  - Gradient accumulation 튜닝
  - 학습률 스케줄링 개선
```

#### 우려
```
⚠️ OOM 위험: 8B 모델은 3B보다 2.7배 크기
```

---

### 13단계: Colab A100 최적화 버전
**커밋**: `f623de9` Add Colab A100 80GB optimized notebook

#### 🚀 최고 성능 환경 구축

**환경 업그레이드**
```
FROM: Kaggle T4 (16GB)
TO:   Colab A100 (80GB)
```

#### 🔥 A100 전용 최적화

**1. 모델 업그레이드**
```
Qwen2.5-VL-3B → Qwen3-VL-8B-Instruct
파라미터: 3B → 8B (2.67배)
```

**2. 성능 설정 극대화**
```python
IMAGE_SIZE: 384 → 768 (4배 픽셀)
BATCH_SIZE: 4 → 16 (4배)
GRAD_ACCUM: 4 → 2 (효율적)
PRECISION: FP16 → BF16 (A100 네이티브)
QUANTIZATION: 4-bit → 8-bit (고품질)
ATTENTION: SDPA → Flash Attention 2 (A100 최적화)
```

**3. LoRA 강화**
```python
LORA_R: 16 → 32 (2배)
LORA_ALPHA: 32 → 64 (2배)
```

#### 예상 성과
```
📊 Single Fold: 83-86%
📊 3-Fold Ensemble: 87-90%
📊 + TTA: 90-93%
⏱️ 학습 시간: ~1.5h/fold
```

#### 핵심 포인트
- **Colab Drive 통합**: /content/drive/MyDrive/kaggle_vqa
- **Output 분리**: ./outputs_a100
- **완전 최적화**: A100의 모든 기능 활용

---

### 14단계: 최종 프롬프트 엔지니어링
**커밋**: `ff19525` Prompt engineering

#### 🔥 마지막 성능 향상

**Problem**: 학습된 모델의 추론 정확도 최대화
```
✅ 해결:
  - 프롬프트 템플릿 정교화
  - 추론 로직 성능 개선
  - 추론 모니터링 강화
```

#### 성과
```
📈 프롬프트 효과: 예상보다 큰 폭 상승
⏰ 제약: 최종 제출 시간 부족으로 더 많은 기능 미적용
```

#### 아쉬움
> "더 많은 실험을 할 시간이 있었다면... 프롬프트 엔지니어링의 잠재력을 확인했지만 시간 제약이 아쉽다."

---

### 15단계: 최종 통합 및 극적인 성능 향상 🏆
**노트북**: `Kaggle_AllInOne_Pro2_2_(2) (1).ipynb`

#### 🚀 최종 돌파구

**핵심 전략**: Qwen3-VL-8B-Instruct 기반 통합 최적화
```
모델: Qwen3-VL-8B-Instruct (FP16)
전략: 모든 최적화 기법 종합 적용
목표: Top 10% 이상 달성
```

#### 🔥 최종 최적화 조합

**1. 라벨 마스킹 완성**
```python
# 프롬프트 토큰 제외, Answer 토큰만 학습
labels = [-100] * len(prompt_tokens) + answer_token_ids
✅ 학습 효율 극대화
```

**2. Direct Logits + 로짓 제한**
```python
# a, b, c, d 토큰만 생성 (추론 안정화)
allowed_tokens = ['a', 'b', 'c', 'd']
logits = model(**inputs).logits[answer_position]
probs = F.softmax(logits[allowed_token_ids], dim=-1)
✅ 추론 안정성 100% 확보
```

**3. 고급 프롬프트 엔지니어링**
```
✅ 질문 유형별 최적화된 템플릿
✅ 명확한 답변 형식 지시
✅ 컨텍스트 강화
```

**4. K-Fold + TTA + Ensemble**
```
✅ Stratified 3-Fold CV
✅ Test-Time Augmentation
✅ 확률 앙상블 (Probability Averaging)
✅ Temperature Scaling (확률 교정)
```

**5. 메모리 및 학습 최적화**
```python
# 자동 GPU VRAM 튜닝
AMP (FP16) + SDPA Attention
Gradient Checkpointing
4-bit QLoRA (r=24~32)
최적 Batch Size + Gradient Accumulation
```

#### 🎯 최종 성과

```
📊 Public Leaderboard Score: 0.92386
📈 베이스라인 대비: +21.5% 향상
📈 중간 점수(0.82716) 대비: +11.7% 향상
🏆 목표(85-88%) 초과 달성!
```

#### 💡 성공의 핵심 요인

**기술적 완성도**
1. **라벨 정렬 + 마스킹**: 학습/추론 완벽 일치
2. **Direct Logits + 로짓 제한**: 100% 안정적 추론
3. **8B 모델 선택**: 성능과 효율의 최적 균형
4. **종합 앙상블**: K-Fold + TTA + 확률 평균

**전략적 선택**
1. 30B 포기 → 8B 집중: 현실적 판단
2. 프롬프트 엔지니어링: 마지막 핵심 개선
3. 모든 최적화 통합: 시너지 효과

#### 교훈
> "각각의 최적화는 작은 향상이지만, 모든 기법을 체계적으로 결합하면 극적인 성능 향상을 이룰 수 있다."

---

## 📊 최종 성과 요약

### 정량적 성과
```
베이스라인:        0.76028
AllInOne:          0.80452 (+5.8%)
AllInOne Pro:      0.82716 (+8.8%)
최종 버전:         0.92386 (+21.5%) 🏆
총 향상:           +21.5% (목표 초과 달성!)
```

### 아키텍처 진화
```
Phase 1: 분산 스크립트 (16개 Python 파일)
Phase 2: 통합 노트북 (All-in-One)
Phase 3: 실험 버전 관리 (experiments/)
Phase 4: 환경별 최적화 (Kaggle T4 / Colab A100)
```

### 모델 진화
```
시도 1: Qwen2.5-VL-3B (베이스라인)
시도 2: Qwen3-VL-30B (실패 - 하드웨어 한계)
시도 3: Qwen3-VL-30B-FP8 (실패 - 복잡도 과다)
최종:   Qwen3-VL-8B (성공 - 최적 밸런스)
```

---

## 🎓 핵심 학습 사항

### 1. T4 GPU 최적화 마스터리
```
✅ Float16 필수 (BFloat16 비효율)
✅ SDPA Attention (FlashAttention 불가)
✅ 4-bit QLoRA (메모리 효율)
✅ Gradient Checkpointing (필수)
```

### 2. 라벨 정렬의 중요성
```
💡 가장 중요한 발견:
   학습 시 Assistant 메시지에 정답 포함
   add_generation_prompt=False 사용
   → 학습/추론 일관성 확보
```

### 3. Direct Logits vs Generate
```
Generate(): 비결정론적, 느림
Direct Logits: 결정론적, 빠름, 안정적
→ 경쟁에서는 Direct Logits 필수
```

### 4. 하드웨어 제약의 현실
```
❌ 30B 모델 실패 교훈:
   - 큰 모델 ≠ 높은 성능
   - 하드웨어 제약 고려 필수
   - 중간 크기 모델 최적화가 현실적
```

### 5. 프롬프트 엔지니어링의 위력
```
📈 마지막 순간 프롬프트 개선으로 큰 폭 상승
💡 초기부터 프롬프트에 투자했다면 더 큰 성과
```

### 6. 실험 관리의 중요성
```
✅ 단일 노트북 워크플로우
✅ experiments/ 폴더 버전 관리
✅ REPORT.md로 변경사항 추적
→ 체계적 실험 관리가 성공의 열쇠
```

---

## 🔮 미래 개선 방향

### 단기 (실현 가능)
1. **Prompt Engineering 심화**: 질문 유형별 프롬프트 최적화
2. **Ensemble 고도화**: Weighted voting, Stacking
3. **Data Augmentation**: Choice shuffle, Paraphrase
4. **Hyperparameter Tuning**: Optuna 적용

### 중기 (도전적)
1. **Knowledge Distillation**: 8B → 3B (속도 향상)
2. **Retrieval-Augmented**: External knowledge 활용
3. **Multi-Task Learning**: Related tasks로 pre-training
4. **Error Analysis**: 실패 케이스 집중 분석

### 장기 (연구적)
1. **Custom Architecture**: VQA 특화 모델 설계
2. **Self-Training**: Pseudo-labeling 활용
3. **Active Learning**: Hard samples 집중 학습

---

## 💎 프로젝트의 가치

### 기술적 성과
- ✅ **Production-Ready 파이프라인**: 에러 핸들링, 로깅, 체크포인트
- ✅ **완전한 문서화**: 15+ 마크다운 문서, 상세 가이드
- ✅ **재사용 가능 코드**: 모듈화된 함수, 클래스
- ✅ **환경별 최적화**: T4 / A100 전용 설정

### 문제 해결 능력
- 🔥 **30B 모델 실패 → 8B 전환**: 현실적 판단
- 🔥 **라벨 정렬 발견**: 핵심 문제 인식 및 해결
- 🔥 **Direct Logits 도입**: 생성 대신 확률 직접 계산
- 🔥 **Multi-GPU 구현**: 병렬 처리 최적화

### 성장 궤적
```
Day 1: T4 호환성 확보 → 기반 구축 (0.76)
Day 2: 통합 노트북 전환 → 실험 효율화 (0.80)
Day 3: Multi-GPU 최적화 → 점진적 향상 (0.83)
Day 4: 대규모 모델 도전 → 한계 인식 및 8B 전환
Day 5: 최종 통합 최적화 → 극적인 돌파 (0.92) 🏆
```

---

## 🏆 결론

이 프로젝트는 단순히 점수 향상을 넘어서, **체계적인 문제 해결 과정**을 보여줍니다:

1. **기반 구축**: T4 호환성, 라벨 정렬 (가장 중요)
2. **반복 실험**: 통합 노트북, 버전 관리
3. **과감한 시도**: 30B 모델 실험 (실패했지만 배움)
4. **현실적 선택**: 8B 모델로 전환
5. **최종 통합**: 모든 최적화 기법 종합 적용 → 극적 성과

> **핵심 교훈**: "각각의 최적화는 작은 향상이지만, 체계적으로 결합하면 극적인 시너지를 만들어낸다. 완벽한 해결책은 없지만, 끊임없는 개선이 탁월한 결과를 만든다."

**최종 성과**: 0.76028 → 0.92386 (+21.5%) 🏆
**목표 달성**: 85-88% 목표 → 92.4% 달성 (목표 초과!)
**진짜 성과**: VQA 시스템 구축 전문성, 고급 트러블슈팅 능력, 체계적 프로젝트 관리 역량

---

**작성일**: 2025-10-27
**프로젝트**: SSAFY AI Project 2025
**레포지토리**: SSAFY_AI_PJT_2025
