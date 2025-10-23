# 📋 Kaggle VQA 명세서 검증 결과 및 최종 프롬프트 사용 가이드

## 🔍 검증 요약

GPT 검증 결과, **6가지 치명적 이슈**와 **다수의 고효율 개선안**이 발견되었으며, 모두 수정 완료했습니다.

---

## ⚠️ 발견된 치명적 이슈 (Critical Blocking Issues)

### 1. Transformers 버전 & 클래스명 오류 ❌→✅
- **문제**: `Qwen2VLForConditionalGeneration`, `Qwen2VLProcessor` 사용
- **원인**: Qwen2.5-VL은 최신 transformers (>=4.49.0) 필요
- **해결**: 
  ```python
  # ✅ 올바른 방식
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
  from qwen_vl_utils import process_vision_info
  
  # 설치
  pip install git+https://github.com/huggingface/transformers.git
  pip install qwen-vl-utils[decord]==0.0.8
  ```
- **영향**: KeyError: 'qwen2_5_vl' → 모델 로드 불가
- **참고**: [HuggingFace 공식 문서](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

### 2. T4 GPU는 BFloat16 미지원 ❌→✅
- **문제**: `bnb_4bit_compute_dtype=torch.bfloat16` 사용
- **원인**: T4는 Turing (SM75) 아키텍처로 BF16 미지원 (Ampere SM80+ 필요)
- **해결**:
  ```python
  # ✅ 올바른 설정
  bnb_config = BitsAndBytesConfig(
      bnb_4bit_compute_dtype=torch.float16  # FP16 사용
  )
  
  training_args = TrainingArguments(
      fp16=True,   # ✅
      bf16=False,  # ✅ T4 미지원
  )
  ```
- **영향**: "Bfloat16 is only supported on GPUs with compute capability of at least 8.0" 에러
- **참고**: [vLLM Issue #1157](https://github.com/vllm-project/vllm/issues/1157)

### 3. FlashAttention 2는 T4 미지원 ❌→✅
- **문제**: `flash-attn==2.6.3`, `attn_implementation="flash_attention_2"` 사용
- **원인**: FA2는 Ampere 이상에서만 최적화됨
- **해결**:
  ```python
  # ✅ 올바른 설정
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_id,
      attn_implementation="sdpa"  # ✅ SDPA 사용 (기본)
  )
  
  # requirements.txt에서 제거
  # flash-attn==2.6.3  ❌ 제거
  ```
- **영향**: 경고 메시지, 성능 향상 없음

### 4. 학습 라벨 정렬 오류 (가장 중요!) ❌→✅
- **문제**: 입력 마지막 토큰 위치에 라벨 설정
- **원인**: HF causal-LM 내부 시프트로 인한 예측 위치 불일치
- **해결**:
  ```python
  # ✅ 올바른 방식: assistant 메시지 포함
  messages = [
      {"role": "user", "content": [...]},
      {"role": "assistant", "content": [{"type": "text", "text": "a"}]}  # ✅ 정답 포함
  ]
  
  text = processor.apply_chat_template(
      messages,
      add_generation_prompt=False  # ✅ False!
  )
  
  # 정답 토큰 위치만 라벨 설정
  labels.fill_(-100)
  labels[0, -len(answer_ids):] = torch.tensor(answer_ids)
  ```
- **영향**: 학습/추론 불일치 → 정확도 크게 감소 (예상 10-15pt)
- **참고**: [Qwen2.5-VL GitHub Issue #709](https://github.com/QwenLM/Qwen2.5-VL/issues/709)

### 5. 수동 특수토큰 구성 금지 ❌→✅
- **문제**: `<|vision_start|>` 등 문자열 직접 조립
- **원인**: 모델/버전 변경 시 깨짐
- **해결**:
  ```python
  # ✅ 올바른 방식
  from qwen_vl_utils import process_vision_info
  
  text = processor.apply_chat_template(messages, tokenize=False)
  images, videos = process_vision_info(messages)
  inputs = processor(text=[text], images=images, videos=videos)
  ```
- **영향**: 버전 업데이트 시 호환성 문제

### 6. 해상도 관리 통일 필요
- **문제**: 학습/추론 해상도 불일치
- **해결**:
  ```python
  # ✅ 올바른 방식
  processor = AutoProcessor.from_pretrained(
      model_id,
      min_pixels=256*28*28,   # 최소 해상도
      max_pixels=768*28*28    # 최대 해상도 (재추론 시 1024~1280)
  )
  ```

---

## 💡 고효율 개선안 (High-Impact Improvements)

### 1. Label Smoothing 추가
```python
training_args = TrainingArguments(
    label_smoothing_factor=0.05,  # 오답 완화
)
```
- **효과**: 과적합 방지, +0.5-1pt 예상

### 2. 확률 평균 앙상블
```python
# ❌ 기존: 로그 확률 가중 합
# ✅ 개선: 확률 평균 (더 안정적)
votes = [answer] * int(weight * 100)
final_answer = Counter(votes).most_common(1)[0][0]
```
- **효과**: 앙상블 안정성 향상, +0.2-0.5pt 예상

### 3. OCR-aware TTA
```python
# OCR 질문은 flip/회전 금지 (문자 반전)
if question_type not in ['ocr']:
    aug_img = augment_image(image_path)
```
- **효과**: OCR 정확도 유지

### 4. Seed 고정 (재현성)
```python
training_args = TrainingArguments(
    seed=42,
    data_seed=42,
)
torch.backends.cudnn.deterministic = True
```

### 5. 자잘한 버그 수정
- `eda.py`: `import re` 누락 추가
- `evaluate.py`: `confusion_matrix` 인자 전달 누락 수정
- `augment.py`: OCR 질문 이미지 증강 제외
- CV: Seed 고정 추가

---

## 📦 생성된 파일 (모두 /mnt/user-data/outputs/)

### 1. [FINAL_CORRECTED_Implementation_Prompt.md](computer:///mnt/user-data/outputs/FINAL_CORRECTED_Implementation_Prompt.md) ⭐ **메인 프롬프트**
**코드 생성형 AI가 바로 사용 가능한 최종 프롬프트**

#### 특징:
- ✅ 6가지 치명적 이슈 모두 수정
- ✅ Phase 0~7 순차 구현 가이드
- ✅ 수정된 코드 예시 포함
- ✅ T4 호환성 보장
- ✅ Completion Criteria 명시

#### 구조:
```
⚠️ CRITICAL FIXES (최우선)
  - 6가지 치명적 이슈 요약
  
🎯 ROLE & MISSION
  
📐 PROJECT CONTEXT
  
🏗️ IMPLEMENTATION PHASES
  ✅ PHASE 0: Project Setup (T4 호환 requirements.txt)
  ✅ PHASE 1: EDA & Preprocessing (import re 추가)
  ✅ PHASE 2: Augmentation (OCR TTA 제외)
  ✅ PHASE 3: Training (라벨 정렬 교정, FP16)  ⭐ 가장 중요
  ✅ PHASE 4: Inference (프롬프트 통일)
  ✅ PHASE 5: Ensemble (확률 평균)
  ✅ PHASE 6: HP Optimization
  ✅ PHASE 7: Final Submission
  
📋 FINAL QUALITY CHECKLIST
  - Critical Fixes 확인
  - Code Quality
  - Reproducibility
```

### 2. [VQA_Specification_Enhancement.md](computer:///mnt/user-data/outputs/VQA_Specification_Enhancement.md)
명세서 개선 보충 사항 (프롬프트 템플릿, 에러 처리 등)

### 3. [Final_Implementation_Prompt.md](computer:///mnt/user-data/outputs/Final_Implementation_Prompt.md)
초기 버전 (치명적 이슈 수정 전)

---

## 🚀 사용 방법

### Step 1: 최종 프롬프트 복사
```bash
# 파일 다운로드
computer:///mnt/user-data/outputs/FINAL_CORRECTED_Implementation_Prompt.md
```

### Step 2: 코드 생성형 AI에 입력
**GPT-4, Claude Sonnet, 또는 기타 코드 생성 AI에 전체 프롬프트 입력**

### Step 3: Phase별 순차 실행
```
사용자: "PHASE 0: Project Setup을 시작합니다. 
디렉토리 구조를 생성하고, T4 호환 requirements.txt를 작성하십시오."

AI: [디렉토리 생성, requirements.txt 작성...]

사용자: "PHASE 1: Data Analysis를 시작합니다."

AI: [EDA 스크립트, Normalizer, StratifiedCV 구현...]

... (Phase 7까지 반복)
```

### Step 4: 각 Phase 검증
```bash
# Phase 0 완료 후
ls -la project/
cat project/requirements.txt | grep "qwen-vl-utils"  # ✅ 확인

# Phase 3 완료 후 (가장 중요!)
python scripts/train_lora.py --fold 0
# ✅ 확인사항:
# - GPU 메모리 < 13GB
# - FP16 사용 확인
# - assistant 메시지 포함 확인
```

---

## ⚙️ 핵심 검증 포인트

프롬프트로 생성된 코드가 다음을 만족하는지 **반드시 확인**하십시오:

### 1. Import 문 검증
```python
# ✅ 올바른 import
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ❌ 잘못된 import
from transformers import Qwen2VLForConditionalGeneration  # 클래스명 오류
```

### 2. BitsAndBytes Config 검증
```python
# ✅ T4 호환
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.float16  # ✅ FP16
)

# ❌ T4 미호환
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.bfloat16  # ❌ BF16
)
```

### 3. Training Arguments 검증
```python
# ✅ T4 호환
TrainingArguments(
    fp16=True,   # ✅
    bf16=False,  # ✅
)
```

### 4. Dataset 라벨 정렬 검증 (가장 중요!)
```python
# ✅ 올바른 방식
messages.append({
    "role": "assistant",
    "content": [{"type": "text", "text": answer}]  # ✅ 정답 포함
})
text = processor.apply_chat_template(
    messages,
    add_generation_prompt=False  # ✅ False!
)

# ❌ 잘못된 방식
text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True  # ❌ True (추론용)
)
# assistant 메시지 없음 ❌
```

### 5. requirements.txt 검증
```txt
# ✅ 필수 포함
transformers>=4.49.0  # 또는 git install
qwen-vl-utils[decord]==0.0.8

# ❌ 제거 확인
# flash-attn==2.6.3  ❌ T4 미지원
```

---

## 📊 예상 성능 향상

| 수정 사항 | 영향도 | 예상 향상 |
|----------|-------|----------|
| 라벨 정렬 교정 | 🔥 Critical | +10-15pt |
| T4 BF16→FP16 | 🔥 Blocking | 학습 가능 |
| Label Smoothing | ⚡ High | +0.5-1pt |
| 확률 평균 앙상블 | ⚡ High | +0.2-0.5pt |
| OCR-aware TTA | ⚡ Medium | +0.3-0.5pt |
| Seed 고정 | ⚡ Medium | 재현성 보장 |

**총 예상 향상: +11-17pt** (라벨 정렬 교정이 핵심)

---

## 🔍 트러블슈팅

### 문제 1: ImportError: cannot import 'Qwen2_5_VLForConditionalGeneration'
**해결**:
```bash
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers.git
```

### 문제 2: ValueError: Bfloat16 is only supported...
**해결**: `torch.float16` 사용 확인
```python
# 모든 코드에서 bfloat16 제거
grep -r "bfloat16" project/  # ❌ 없어야 함
grep -r "torch.float16" project/  # ✅ 있어야 함
```

### 문제 3: Validation accuracy가 25% 근처 (random guess)
**원인**: 라벨 정렬 오류
**해결**: assistant 메시지 포함 확인
```python
# scripts/train_lora.py 확인
# messages에 assistant 응답이 있는지 확인
messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
```

### 문제 4: GPU Out of Memory
**해결**:
```python
# batch_size 줄이기
per_device_train_batch_size=2  # 4 → 2
gradient_accumulation_steps=4  # 2 → 4
```

---

## 📌 최종 체크리스트

프로젝트 완성 후 다음을 확인하십시오:

- [ ] ✅ **Transformers Git Install**: `pip list | grep transformers` → dev 버전 확인
- [ ] ✅ **qwen-vl-utils 설치**: `pip list | grep qwen-vl-utils` → 0.0.8
- [ ] ✅ **클래스명**: 모든 코드에서 `Qwen2_5_VL*` 사용
- [ ] ✅ **FP16**: 모든 코드에서 `torch.float16` 사용, `bfloat16` 없음
- [ ] ✅ **FlashAttention 제거**: requirements.txt, 코드에서 모두 제거
- [ ] ✅ **라벨 정렬**: train_lora.py에서 assistant 메시지 포함 확인
- [ ] ✅ **프롬프트 통일**: apply_chat_template + process_vision_info 사용
- [ ] ✅ **해상도 관리**: min_pixels/max_pixels 설정 확인
- [ ] ✅ **Seed 고정**: seed=42, deterministic=True 설정
- [ ] ✅ **제출 파일 검증**: validate_submission.py 통과

---

## 🎉 결론

**6가지 치명적 이슈**를 모두 수정한 최종 프롬프트가 준비되었습니다.

**특히 "라벨 정렬 교정" (Issue #4)이 가장 중요**하며, 이것만으로도 **+10-15pt 향상**이 예상됩니다.

프롬프트를 코드 생성 AI에 입력하고, 각 Phase를 순차적으로 진행하면서 위 체크리스트를 확인하십시오.

**성공을 기원합니다! 🚀**
