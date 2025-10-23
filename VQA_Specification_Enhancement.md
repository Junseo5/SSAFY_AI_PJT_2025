# Kaggle VQA Challenge - 명세서 개선 보충 사항

## 🔍 추가된 핵심 개선 사항

### 1. 프롬프트 엔지니어링 전략 (신규)

#### 1.1 질문 유형별 최적화 프롬프트

```python
# config/prompt_templates.yaml
prompt_templates:
  counting:
    system: "You are a visual counting expert. Analyze the image carefully and count objects precisely."
    user: |
      Question: {question}
      
      Choices:
      a) {choice_a}
      b) {choice_b}
      c) {choice_c}
      d) {choice_d}
      
      Instructions:
      1. Locate all relevant objects in the image
      2. Count each object carefully
      3. Match your count with the closest choice
      4. Answer ONLY with a single letter: a, b, c, or d
      
  color:
    system: "You are a color recognition expert. Identify colors accurately considering lighting and context."
    user: |
      Question: {question}
      
      Choices:
      a) {choice_a}
      b) {choice_b}
      c) {choice_c}
      d) {choice_d}
      
      Instructions:
      1. Identify the primary color of the specified object
      2. Consider lighting conditions
      3. Select the most accurate color description
      4. Answer ONLY with a single letter: a, b, c, or d
      
  ocr:
    system: "You are an OCR specialist. Extract text accurately from images."
    user: |
      Question: {question}
      
      Choices:
      a) {choice_a}
      b) {choice_b}
      c) {choice_c}
      d) {choice_d}
      
      Instructions:
      1. Locate text in the image
      2. Read it carefully (supports Korean, English, numbers)
      3. Match with the provided choices
      4. Answer ONLY with a single letter: a, b, c, or d
      
  yesno:
    system: "You are a visual reasoning expert. Determine if statements are true or false based on the image."
    user: |
      Question: {question}
      
      Choices:
      a) {choice_a}
      b) {choice_b}
      c) {choice_c}
      d) {choice_d}
      
      Instructions:
      1. Verify the statement against image content
      2. Answer with the appropriate yes/no/correct/incorrect option
      3. Answer ONLY with a single letter: a, b, c, or d
      
  general:
    system: "You are a visual question answering expert. Analyze images and answer questions accurately."
    user: |
      Question: {question}
      
      Choices:
      a) {choice_a}
      b) {choice_b}
      c) {choice_c}
      d) {choice_d}
      
      Instructions:
      1. Carefully analyze the image
      2. Consider all provided choices
      3. Select the most accurate answer
      4. Answer ONLY with a single letter: a, b, c, or d
```

#### 1.2 프롬프트 적용 로직

```python
# scripts/prompt_manager.py
import yaml

class PromptManager:
    def __init__(self, templates_path='config/prompt_templates.yaml'):
        with open(templates_path, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)['prompt_templates']
    
    def format_prompt(self, question_type, question, choices):
        """질문 유형에 맞는 프롬프트 생성"""
        template = self.templates.get(question_type, self.templates['general'])
        
        return {
            'system': template['system'],
            'user': template['user'].format(
                question=question,
                choice_a=choices['a'],
                choice_b=choices['b'],
                choice_c=choices['c'],
                choice_d=choices['d']
            )
        }
    
    def apply_few_shot(self, prompt, question_type, n_shots=2):
        """Few-shot learning 적용"""
        examples = self._get_examples(question_type, n_shots)
        
        few_shot_prefix = "\n\nHere are some examples:\n"
        for i, ex in enumerate(examples, 1):
            few_shot_prefix += f"\nExample {i}:\n"
            few_shot_prefix += f"Question: {ex['question']}\n"
            few_shot_prefix += f"Answer: {ex['answer']}\n"
        
        prompt['user'] = few_shot_prefix + "\n\nNow answer this:\n" + prompt['user']
        return prompt
    
    def _get_examples(self, question_type, n=2):
        """유형별 예시 샘플 (하드코딩 또는 DB에서 로드)"""
        examples_db = {
            'counting': [
                {'question': '이미지에 사과가 몇 개 있습니까?', 'answer': 'c'},
                {'question': 'How many cars are in the parking lot?', 'answer': 'b'}
            ],
            'color': [
                {'question': '이 차는 무슨 색입니까?', 'answer': 'a'},
                {'question': 'What color is the sky?', 'answer': 'd'}
            ],
            # ... 다른 유형들
        }
        return examples_db.get(question_type, [])[:n]
```

---

### 2. 체계적 에러 처리 전략

```python
# scripts/error_handler.py
import logging
import torch
from typing import Callable, Any
from functools import wraps

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VQAErrorHandler:
    """VQA 프로젝트 전용 에러 핸들러"""
    
    @staticmethod
    def handle_gpu_oom(func: Callable) -> Callable:
        """GPU OOM 자동 복구"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ GPU OOM detected. Clearing cache...")
                    torch.cuda.empty_cache()
                    
                    # Batch size 줄이기
                    if 'batch_size' in kwargs:
                        new_batch_size = max(1, kwargs['batch_size'] // 2)
                        logger.info(f"🔄 Reducing batch_size: {kwargs['batch_size']} → {new_batch_size}")
                        kwargs['batch_size'] = new_batch_size
                        return func(*args, **kwargs)
                    else:
                        raise
                else:
                    raise
        return wrapper
    
    @staticmethod
    def handle_tokenization_error(func: Callable) -> Callable:
        """한글 토큰화 오류 방지"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "encode" in str(e).lower() or "token" in str(e).lower():
                    logger.error(f"❌ Tokenization failed: {e}")
                    
                    # 텍스트 정규화 재시도
                    if 'text' in kwargs:
                        import unicodedata
                        kwargs['text'] = unicodedata.normalize('NFKC', kwargs['text'])
                        logger.info("🔄 Retrying with normalized text...")
                        return func(*args, **kwargs)
                    else:
                        raise
                else:
                    raise
        return wrapper
    
    @staticmethod
    def handle_model_load_error(func: Callable) -> Callable:
        """모델 로딩 실패 처리"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "load" in str(e).lower() or "checkpoint" in str(e).lower():
                    logger.error(f"❌ Model loading failed: {e}")
                    
                    # 백업 체크포인트 시도
                    if 'model_path' in kwargs:
                        backup_path = kwargs['model_path'].replace('fold0', 'fold1')
                        if backup_path != kwargs['model_path']:
                            logger.info(f"🔄 Trying backup checkpoint: {backup_path}")
                            kwargs['model_path'] = backup_path
                            return func(*args, **kwargs)
                    
                    raise
                else:
                    raise
        return wrapper
    
    @staticmethod
    def safe_inference(predictor, image_path, question, choices, max_retries=3):
        """안전한 추론 (재시도 포함)"""
        for attempt in range(max_retries):
            try:
                result = predictor.predict(image_path, question, choices)
                
                # 결과 검증
                if result['prediction'] not in ['a', 'b', 'c', 'd']:
                    raise ValueError(f"Invalid prediction: {result['prediction']}")
                
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ Inference attempt {attempt+1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"❌ All retries failed for image: {image_path}")
                    # 폴백: 무작위 답 반환
                    import random
                    fallback = random.choice(['a', 'b', 'c', 'd'])
                    logger.warning(f"🎲 Using fallback prediction: {fallback}")
                    return {'prediction': fallback, 'confidence': 0.0}
                
                # 재시도 전 대기
                import time
                time.sleep(1)

# 사용 예시
@VQAErrorHandler.handle_gpu_oom
@VQAErrorHandler.handle_tokenization_error
def train_epoch(model, dataloader, optimizer, batch_size=4):
    # 학습 코드
    pass
```

---

### 3. 하이퍼파라미터 탐색 방법론

```python
# scripts/hyperparameter_search.py
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb

class HyperparameterOptimizer:
    def __init__(self, study_name='vqa-optimization'):
        self.study_name = study_name
        
    def objective(self, trial):
        """Optuna 목적 함수"""
        # 하이퍼파라미터 샘플링
        config = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 5e-5),
            'lora_r': trial.suggest_categorical('lora_r', [16, 24, 32]),
            'lora_alpha': trial.suggest_categorical('lora_alpha', [32, 48, 64]),
            'num_epochs': trial.suggest_int('num_epochs', 3, 5),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-3, 1e-1),
            'warmup_ratio': trial.suggest_uniform('warmup_ratio', 0.0, 0.1),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4])
        }
        
        # 학습 실행
        accuracy = self._train_and_evaluate(config)
        
        # WandB 로깅
        wandb.log({
            'trial_number': trial.number,
            'accuracy': accuracy,
            **config
        })
        
        return accuracy
    
    def _train_and_evaluate(self, config):
        """설정으로 학습 후 검증 정확도 반환"""
        # 실제 학습 코드 호출
        # return val_accuracy
        pass
    
    def optimize(self, n_trials=20):
        """최적화 실행"""
        wandb_callback = WeightsAndBiasesCallback(
            wandb_kwargs={'project': 'kaggle-vqa-optimization'}
        )
        
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()  # 나쁜 시도 조기 종료
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[wandb_callback]
        )
        
        # 최적 하이퍼파라미터
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"✅ Best hyperparameters: {best_params}")
        print(f"✅ Best accuracy: {best_value:.4f}")
        
        return best_params

# 사용법
if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    best_config = optimizer.optimize(n_trials=15)  # Day 4에 실행
    
    # 최적 설정 저장
    import yaml
    with open('config/best_config.yaml', 'w') as f:
        yaml.dump(best_config, f)
```

---

### 4. Cross-Validation 전략 상세화

```python
# scripts/stratified_cv.py
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

class VQAStratifiedSplitter:
    def __init__(self, n_folds=3, seed=42):
        self.n_folds = n_folds
        self.seed = seed
    
    def create_folds(self, df):
        """질문 유형 비율을 유지하는 Stratified K-Fold"""
        # 질문 유형 분류
        df = self._classify_questions(df)
        
        # 복합 stratify 레이블 생성 (유형 + 정답)
        df['stratify_label'] = df['question_type'] + '_' + df['answer']
        
        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed
        )
        
        df['fold'] = -1
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(df, df['stratify_label'])
        ):
            df.loc[val_idx, 'fold'] = fold_idx
        
        # 분포 확인
        self._print_fold_distribution(df)
        
        return df
    
    def _classify_questions(self, df):
        """질문 유형 자동 분류"""
        import re
        
        def classify(question):
            patterns = {
                'counting': r'몇|개수|수|how many',
                'color': r'색|색깔|color|무슨색',
                'ocr': r'글자|문자|숫자|번호|읽|text|number',
                'yesno': r'인가|입니까|\?$|있는가|맞는가',
                'location': r'어디|위치|where|장소',
                'attribute': r'무엇|what|어떤|kind'
            }
            
            for qtype, pattern in patterns.items():
                if re.search(pattern, question, re.I):
                    return qtype
            return 'general'
        
        df['question_type'] = df['question'].apply(classify)
        return df
    
    def _print_fold_distribution(self, df):
        """Fold별 분포 출력"""
        print("\n📊 Fold Distribution:")
        print("=" * 60)
        
        for fold in range(self.n_folds):
            fold_df = df[df['fold'] == fold]
            print(f"\nFold {fold} ({len(fold_df)} samples):")
            
            # 질문 유형 분포
            type_dist = fold_df['question_type'].value_counts()
            for qtype, count in type_dist.items():
                pct = count / len(fold_df) * 100
                print(f"  {qtype:12s}: {count:4d} ({pct:5.1f}%)")
            
            # 정답 분포
            answer_dist = fold_df['answer'].value_counts()
            print(f"  Answers: {dict(answer_dist)}")

# 사용 예시
if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    
    splitter = VQAStratifiedSplitter(n_folds=3, seed=42)
    train_df = splitter.create_folds(train_df)
    
    train_df.to_csv('data/train_with_folds.csv', index=False)
```

---

### 5. GPU 메모리 관리 최적화

```python
# scripts/memory_optimizer.py
import torch
import gc

class GPUMemoryManager:
    def __init__(self):
        self.peak_memory = 0
    
    @staticmethod
    def clear_cache():
        """GPU 캐시 정리"""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_stats():
        """메모리 사용 통계"""
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
        return stats
    
    @staticmethod
    def optimize_training_config(available_memory_gb=15):
        """사용 가능 메모리에 따른 최적 설정"""
        configs = {
            15: {  # T4 GPU
                'batch_size': 4,
                'gradient_accumulation_steps': 2,
                'max_seq_length': 512,
                'use_gradient_checkpointing': True,
                'mixed_precision': 'bf16'
            },
            30: {  # T4 x2
                'batch_size': 8,
                'gradient_accumulation_steps': 1,
                'max_seq_length': 1024,
                'use_gradient_checkpointing': False,
                'mixed_precision': 'bf16'
            }
        }
        
        # 가장 가까운 설정 선택
        available_configs = [k for k in configs.keys() if k <= available_memory_gb]
        if available_configs:
            selected = max(available_configs)
            return configs[selected]
        else:
            return configs[15]  # 기본값
    
    def monitor_training(self, callback_interval=100):
        """학습 중 메모리 모니터링"""
        def callback(step):
            if step % callback_interval == 0:
                stats = self.get_memory_stats()
                self.peak_memory = max(self.peak_memory, stats['allocated'])
                
                print(f"Step {step} - Memory: {stats['allocated']:.2f}GB / {stats['reserved']:.2f}GB")
                
                # 메모리 경고
                if stats['allocated'] > 13:  # 15GB의 87%
                    print("⚠️ WARNING: Memory usage is high!")
                    self.clear_cache()
        
        return callback

# Training Arguments에 적용
from transformers import TrainingArguments

def create_memory_efficient_training_args(output_dir='checkpoints'):
    """메모리 효율적인 학습 설정"""
    manager = GPUMemoryManager()
    config = manager.optimize_training_config(available_memory_gb=15)
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        gradient_checkpointing=config['use_gradient_checkpointing'],
        fp16=False,
        bf16=True if config['mixed_precision'] == 'bf16' else False,
        optim="paged_adamw_8bit",  # 메모리 효율적 옵티마이저
        dataloader_pin_memory=False,  # 메모리 절약
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # 체크포인트 2개만 유지
    )
```

---

### 6. 실험 추적 체계

```python
# scripts/experiment_tracker.py
import wandb
import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, project_name='kaggle-vqa', entity=None):
        self.project_name = project_name
        self.entity = entity
        self.experiment_log = []
    
    def start_experiment(self, config, experiment_name=None):
        """실험 시작"""
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            tags=self._generate_tags(config)
        )
        
        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        return wandb.run
    
    def log_metrics(self, metrics, step=None):
        """메트릭 로깅"""
        wandb.log(metrics, step=step)
    
    def log_model_artifact(self, model_path, name, metadata=None):
        """모델 체크포인트 업로드"""
        artifact = wandb.Artifact(
            name=name,
            type='model',
            metadata=metadata
        )
        artifact.add_dir(model_path)
        wandb.log_artifact(artifact)
    
    def finish_experiment(self, final_metrics):
        """실험 종료"""
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['final_metrics'] = final_metrics
        self.current_experiment['status'] = 'completed'
        
        self.experiment_log.append(self.current_experiment)
        
        # 로컬에 저장
        log_path = Path('experiments/log.json')
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        wandb.finish()
    
    def compare_experiments(self, experiment_names):
        """실험 비교"""
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project_name}")
        
        comparison = []
        for run in runs:
            if run.name in experiment_names:
                comparison.append({
                    'name': run.name,
                    'config': run.config,
                    'summary': run.summary._json_dict
                })
        
        return comparison
    
    def _generate_tags(self, config):
        """설정 기반 태그 생성"""
        tags = []
        
        if 'model_id' in config:
            if '7B' in config['model_id']:
                tags.append('7B')
            elif '3B' in config['model_id']:
                tags.append('3B')
        
        if 'fold' in config:
            tags.append(f"fold{config['fold']}")
        
        if 'learning_rate' in config:
            lr_tag = f"lr{config['learning_rate']:.0e}"
            tags.append(lr_tag)
        
        return tags

# 사용 예시
tracker = ExperimentTracker(project_name='kaggle-vqa')

config = {
    'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'fold': 0,
    'learning_rate': 2e-5,
    'lora_r': 24,
    'num_epochs': 3
}

run = tracker.start_experiment(config, experiment_name='7b_fold0_v1')

# 학습...
tracker.log_metrics({'train_loss': 0.35, 'val_accuracy': 0.82})

# 완료
tracker.finish_experiment({'final_accuracy': 0.845})
```

---

### 7. 디버깅 가이드

```markdown
## 🔧 Debugging Guide

### Common Issues & Solutions

#### Issue 1: GPU Out of Memory
**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결 방법:**
1. Batch size 줄이기: `per_device_train_batch_size=2`
2. Gradient accumulation 증가: `gradient_accumulation_steps=4`
3. Gradient checkpointing 활성화: `gradient_checkpointing=True`
4. Mixed precision 사용: `bf16=True`

```python
# scripts/debug_memory.py
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
torch.cuda.empty_cache()
```

---

#### Issue 2: 한글 토큰화 오류
**증상:**
```
UnicodeEncodeError: 'utf-8' codec can't encode...
```

**해결 방법:**
```python
import unicodedata

def safe_tokenize(text):
    # NFKC 정규화
    text = unicodedata.normalize('NFKC', text)
    
    # 특수문자 제거
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return tokenizer(text, truncation=True, max_length=512)
```

---

#### Issue 3: 모델 예측이 항상 'a'
**증상:**
- 모든 샘플에 대해 동일한 답 예측
- Validation accuracy가 25% 근처

**원인:**
- 프롬프트 형식 오류
- Forced-choice 파싱 실패

**해결 방법:**
```python
# scripts/debug_prediction.py
def debug_single_prediction(predictor, sample):
    # 전체 생성 텍스트 확인
    with torch.no_grad():
        outputs = predictor.model.generate(
            **inputs,
            max_new_tokens=50,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    generated_text = predictor.processor.decode(outputs.sequences[0])
    print(f"Full output: {generated_text}")
    
    # 파싱 과정 확인
    parsed = predictor._parse_answer(generated_text)
    print(f"Parsed answer: {parsed}")
```

---

#### Issue 4: Cross-Validation 점수 불안정
**증상:**
- Fold 0: 85%, Fold 1: 72%, Fold 2: 88%
- 큰 분산

**원인:**
- Fold 분할이 질문 유형 비율을 고려하지 않음
- 특정 Fold에 어려운 샘플 집중

**해결 방법:**
- Stratified K-Fold 사용 (위 Section 4 참고)
- Fold별 질문 유형 분포 확인

---

#### Issue 5: 제출 파일 형식 오류
**증상:**
```
Submission failed: Invalid format
```

**체크리스트:**
```python
# scripts/validate_submission.py
import pandas as pd

def validate_submission(file_path):
    df = pd.read_csv(file_path)
    
    # 1. 컬럼 확인
    assert list(df.columns) == ['id', 'answer'], "Columns must be ['id', 'answer']"
    
    # 2. ID 중복 확인
    assert not df['id'].duplicated().any(), "Duplicate IDs found"
    
    # 3. 답 형식 확인
    assert df['answer'].isin(['a', 'b', 'c', 'd']).all(), "Invalid answers found"
    
    # 4. 공백 확인
    assert not df['answer'].str.contains(' ').any(), "Whitespace in answers"
    
    # 5. 모든 test ID 포함 확인
    test_ids = pd.read_csv('data/test.csv')['id'].values
    assert set(df['id']) == set(test_ids), "Missing or extra IDs"
    
    print("✅ Submission file is valid!")

validate_submission('outputs/submission_final.csv')
```
```

---

## 📝 통합 체크리스트

```markdown
## Day-by-Day Implementation Checklist

### Day 1 (Foundation)
- [ ] Environment setup completed
  - [ ] All packages installed
  - [ ] GPU connectivity verified
  - [ ] WandB login successful
- [ ] EDA & Analysis
  - [ ] Question type distribution analyzed
  - [ ] Normalization rules created
  - [ ] Prompt templates prepared
- [ ] Baseline
  - [ ] Zero-shot inference working
  - [ ] First submission made (65-68%)
- [ ] Data Pipeline
  - [ ] Augmentation scripts tested
  - [ ] Stratified CV splits created
  - [ ] Data loaders validated

### Day 2 (7B Training)
- [ ] Training Setup
  - [ ] 4-bit quantization working
  - [ ] LoRA configuration validated
  - [ ] Memory usage optimized (<13GB)
- [ ] Model Training
  - [ ] Fold 0 completed
  - [ ] Fold 1 completed
  - [ ] Fold 2 running overnight
- [ ] Monitoring
  - [ ] WandB tracking active
  - [ ] Checkpoints saved
  - [ ] CV accuracy logged

### Day 3 (3B + Inference)
- [ ] 3B Training (GPU 1)
  - [ ] 3-fold training completed
  - [ ] Models saved
- [ ] 7B Inference
  - [ ] Forced-choice predictions generated
  - [ ] Submission file validated
  - [ ] 2nd submission made (79-82%)
- [ ] Analysis
  - [ ] 7B vs 3B performance compared
  - [ ] Error patterns identified

### Day 4 (Optimization)
- [ ] Ensemble
  - [ ] 7B 3-fold ensemble implemented
  - [ ] 7B+3B mixed ensemble tested
  - [ ] Margin-based re-inference working
- [ ] Hyperparameter Search
  - [ ] Optuna study completed (10+ trials)
  - [ ] Best config identified
  - [ ] Re-training with best params
- [ ] Advanced Features
  - [ ] OCR pipeline integrated
  - [ ] Type-specific post-processing applied
  - [ ] 3rd submission made (83-85%)

### Day 5 (Final Push)
- [ ] Error Analysis
  - [ ] Top 20 errors analyzed
  - [ ] Weak question types identified
  - [ ] Targeted data augmentation done
- [ ] Final Optimizations
  - [ ] High-resolution (448px) inference
  - [ ] TTA applied
  - [ ] Ensemble weights tuned
- [ ] Submission
  - [ ] 4-5 final submissions made
  - [ ] Best result documented
  - [ ] Target achieved (85-88%)
- [ ] Documentation
  - [ ] README completed
  - [ ] Code cleaned
  - [ ] Presentation prepared
```
