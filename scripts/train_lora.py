"""
Kaggle VQA Challenge - QLoRA Training Script

✅ CRITICAL FIXES APPLIED:
1. Qwen2_5_VLForConditionalGeneration (not Qwen2VL...)
2. AutoProcessor (not Qwen2VLProcessor)
3. torch.float16 (T4 compatible, NOT bfloat16)
4. attn_implementation="sdpa" (FlashAttention 2 removed)
5. qwen_vl_utils.process_vision_info (required)
6. apply_chat_template (standard method)
7. assistant message with answer (label alignment fix)
8. add_generation_prompt=False for training
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,  # ✅ Correct class name
    AutoProcessor,                        # ✅ Correct processor
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info  # ✅ Required import
from PIL import Image
import pandas as pd
import unicodedata
import os
from pathlib import Path
from typing import Dict, List
import numpy as np


class VQADataset(torch.utils.data.Dataset):
    """
    VQA 데이터셋 클래스

    ✅ CRITICAL: 라벨 정렬 교정 - assistant 메시지에 정답 포함
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        processor,
        prompt_manager,
        normalizer
    ):
        """
        Args:
            df: train_with_folds.csv 데이터프레임
            image_dir: 이미지 폴더 경로
            processor: AutoProcessor 인스턴스
            prompt_manager: PromptManager 인스턴스
            normalizer: AnswerNormalizer 인스턴스
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.prompt_manager = prompt_manager
        self.normalizer = normalizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        단일 샘플 반환

        ✅ CRITICAL: 라벨 정렬 교정
        - assistant 메시지에 정답 1글자 포함
        - apply_chat_template(add_generation_prompt=False)
        - 정답 토큰 위치만 학습

        Returns:
            dict: {
                'pixel_values': Tensor,
                'input_ids': Tensor,
                'attention_mask': Tensor,
                'labels': Tensor
            }
        """
        row = self.df.iloc[idx]

        # 이미지 경로
        if 'image' in row:
            image_path = os.path.join(self.image_dir, row['image'])
        else:
            raise ValueError(f"No 'image' column in row {idx}")

        # 이미지 존재 확인
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 질문 유형
        question_type = row.get('question_type', 'general')

        # 보기 구성
        choices = {
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'd': row['d']
        }

        # 정답
        answer = row['answer'].lower().strip()  # 'a', 'b', 'c', 'd'

        # ✅ CRITICAL: 학습용 메시지 생성 (assistant 포함)
        messages = self.prompt_manager.build_training_messages(
            image_path=image_path,
            question_type=question_type,
            question=row['question'],
            choices=choices,
            answer=answer
        )

        # ✅ CRITICAL: apply_chat_template 사용
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # ✅ False for training!
        )

        # ✅ 한글 정규화 (토큰화 오류 방지)
        text = unicodedata.normalize('NFKC', text)

        # ✅ CRITICAL: process_vision_info 사용
        images, videos = process_vision_info(messages)

        # 인코딩
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        )

        # ✅ CRITICAL: 라벨 정렬 교정
        labels = inputs["input_ids"].clone()
        labels.fill_(-100)  # 모든 토큰 무시

        # 정답 토큰 ID 찾기
        answer_ids = self.processor.tokenizer.encode(
            answer,
            add_special_tokens=False
        )

        # 마지막 answer_ids 길이만큼만 라벨 설정
        if len(answer_ids) > 0:
            labels[0, -len(answer_ids):] = torch.tensor(answer_ids)

        return {
            'pixel_values': inputs['pixel_values'][0],
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'labels': labels[0]
        }


def create_model_and_processor(
    model_id: str,
    device: str = "cuda:0",
    lora_r: int = 24,
    lora_alpha: int = 48
):
    """
    모델 및 프로세서 생성

    ✅ CRITICAL FIXES:
    - Qwen2_5_VLForConditionalGeneration
    - AutoProcessor
    - torch.float16 (T4 compatible)
    - attn_implementation="sdpa"
    - FlashAttention 제거

    Args:
        model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
        device: "cuda:0" or "cuda:1"
        lora_r: LoRA rank
        lora_alpha: LoRA alpha

    Returns:
        tuple: (model, processor)
    """
    print(f"\n{'='*60}")
    print("Creating Model and Processor")
    print(f"{'='*60}\n")

    # ✅ CRITICAL: BitsAndBytes Config (FP16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # ✅ FP16, NOT bfloat16
    )

    print(f"Loading model: {model_id}")
    print(f"Quantization: 4-bit NF4")
    print(f"Compute dtype: torch.float16 (T4 compatible)")
    print(f"Attention: SDPA (FlashAttention 2 removed)\n")

    # ✅ CRITICAL: 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.float16,      # ✅ FP16
        attn_implementation="sdpa"       # ✅ SDPA, not flash_attention_2
    )

    print("✓ Model loaded successfully")

    # K-bit training 준비
    model = prepare_model_for_kbit_training(model)

    # ✅ LoRA Config
    print(f"\nConfiguring LoRA:")
    print(f"  r={lora_r}, alpha={lora_alpha}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            # Language model only (Vision encoder frozen)
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # 학습 가능 파라미터 출력
    model.print_trainable_parameters()

    # ✅ CRITICAL: AutoProcessor 로드
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=256*28*28,   # ✅ 해상도 관리
        max_pixels=768*28*28    # ✅ 해상도 관리
    )

    print("✓ Processor loaded successfully")
    print(f"  min_pixels: {256*28*28}")
    print(f"  max_pixels: {768*28*28}")

    return model, processor


def compute_metrics(eval_pred):
    """
    평가 메트릭 계산

    Args:
        eval_pred: (predictions, labels)

    Returns:
        dict: {'accuracy': float}
    """
    predictions, labels = eval_pred

    # Logits → predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    # Labels에서 -100 제외
    mask = labels != -100
    predictions_masked = predictions[mask]
    labels_masked = labels[mask]

    # 정확도
    if len(labels_masked) > 0:
        accuracy = (predictions_masked == labels_masked).mean()
    else:
        accuracy = 0.0

    return {'accuracy': float(accuracy)}


def train(
    model_id: str,
    train_csv: str,
    image_dir: str,
    output_dir: str,
    fold: int = 0,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    lora_r: int = 24,
    lora_alpha: int = 48,
    device: str = "cuda:0"
):
    """
    학습 실행 함수

    ✅ CRITICAL: 모든 치명적 이슈 수정 적용

    Args:
        model_id: 모델 ID
        train_csv: 학습 데이터 CSV (fold 정보 포함)
        image_dir: 이미지 디렉토리
        output_dir: 출력 디렉토리
        fold: Validation fold 번호
        num_epochs: 에폭 수
        learning_rate: 학습률
        batch_size: 배치 크기
        gradient_accumulation_steps: Gradient accumulation
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        device: GPU 디바이스
    """
    print("\n" + "="*60)
    print("VQA QLoRA Training")
    print("="*60 + "\n")

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # 모델 및 프로세서 생성
    model, processor = create_model_and_processor(
        model_id=model_id,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # 유틸리티 초기화
    import sys
    sys.path.append('.')
    from scripts.prompt_manager import PromptManager
    from scripts.normalize import AnswerNormalizer

    prompt_manager = PromptManager()
    normalizer = AnswerNormalizer()

    # 데이터 로드
    print(f"\n{'='*60}")
    print("Loading Data")
    print(f"{'='*60}\n")

    df = pd.read_csv(train_csv)
    print(f"Total samples: {len(df)}")

    # Fold 분할
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    print(f"Fold {fold}:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # 데이터셋 생성
    train_dataset = VQADataset(
        train_df, image_dir, processor, prompt_manager, normalizer
    )
    val_dataset = VQADataset(
        val_df, image_dir, processor, prompt_manager, normalizer
    )

    print(f"\n✓ Datasets created")

    # ✅ CRITICAL: Training Arguments (FP16, label_smoothing, seed)
    print(f"\n{'='*60}")
    print("Creating Training Arguments")
    print(f"{'='*60}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,

        # Batch size
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        # ✅ CRITICAL: T4 compatibility
        fp16=True,              # ✅ FP16
        bf16=False,             # ✅ NOT BF16

        # Optimizer
        optim="paged_adamw_8bit",

        # Learning rate
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # Regularization
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.05,  # ✅ Label smoothing

        # Logging
        logging_steps=10,
        logging_first_step=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=50,

        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",

        # WandB
        report_to="wandb",

        # ✅ CRITICAL: Reproducibility
        seed=42,
        data_seed=42,

        # Disable
        push_to_hub=False,
        dataloader_pin_memory=False,
    )

    print("Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Precision: FP16 (T4 compatible)")
    print(f"  Label smoothing: 0.05")
    print(f"  Seed: 42")

    # ✅ CRITICAL: CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Trainer
    print(f"\n{'='*60}")
    print("Creating Trainer")
    print(f"{'='*60}\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("✓ Trainer created")

    # 학습 시작
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    trainer.train()

    # 저장
    print(f"\n{'='*60}")
    print("Saving Model")
    print(f"{'='*60}\n")

    final_path = os.path.join(output_dir, 'final')
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)

    print(f"✓ Model saved to {final_path}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Train VQA model with QLoRA')
    parser.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model ID')
    parser.add_argument('--train_csv', default='data/train_with_folds.csv', help='Training CSV with folds')
    parser.add_argument('--image_dir', default='data/images', help='Image directory')
    parser.add_argument('--output_dir', default='checkpoints/qwen-7b-fold0', help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Validation fold')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--lora_r', type=int, default=24, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=48, help='LoRA alpha')
    parser.add_argument('--device', default='cuda:0', help='Device')

    args = parser.parse_args()

    # WandB 초기화
    try:
        import wandb
        wandb.init(
            project='kaggle-vqa',
            name=f'7b-fold{args.fold}-fp16',
            config=vars(args)
        )
        print("✓ WandB initialized")
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print("  Continuing without WandB...")

    # 학습 실행
    train(
        model_id=args.model_id,
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        fold=args.fold,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=args.device
    )


if __name__ == "__main__":
    main()
