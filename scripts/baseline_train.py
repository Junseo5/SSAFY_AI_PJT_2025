"""
Kaggle VQA Challenge - Baseline-Style Training Script

This is a simplified training script following the baseline notebook structure.
Perfect for quick testing and prototyping.

Features:
- Simple and straightforward
- Direct path column usage
- Compatible with AutoModelForVision2Seq
- Fast iteration for experiments
"""

import os, re, math, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
from typing import Any
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import argparse

# 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None


# 프롬프트 템플릿
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


def build_mc_prompt(question, a, b, c, d):
    """Multiple choice 프롬프트 생성"""
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


# 커스텀 데이터셋
class VQAMCDataset(Dataset):
    """Baseline 스타일 데이터셋"""

    def __init__(self, df, processor, data_dir="", train=True):
        """
        Args:
            df: DataFrame with 'path' column
            processor: AutoProcessor
            data_dir: Base directory for images (if needed)
            train: Training mode (includes answer)
        """
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.data_dir = data_dir
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # 이미지 경로 처리
        if 'path' in row:
            img_path = os.path.join(self.data_dir, row["path"])
        elif 'image' in row:
            img_path = os.path.join(self.data_dir, row["image"])
        else:
            raise ValueError("No 'path' or 'image' column found")

        # 이미지 로드
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️  Warning: Image not found: {img_path}, using blank image")
            img = Image.new('RGB', (384, 384), color='white')

        # 프롬프트 생성
        q = str(row["question"])
        a, b, c, d = str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])
        user_text = build_mc_prompt(q, a, b, c, d)

        # 메시지 구성
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text}
            ]}
        ]

        # 학습 시 정답 추가
        if self.train and 'answer' in row:
            gold = str(row["answer"]).strip().lower()
            messages.append({"role": "assistant", "content": [{"type": "text", "text": gold}]})

        return {"messages": messages, "image": img}


# 데이터 콜레이터
@dataclass
class DataCollator:
    """Baseline 스타일 콜레이터"""
    processor: Any
    train: bool = True

    def __call__(self, batch):
        texts, images = [], []
        for sample in batch:
            messages = sample["messages"]
            img = sample["image"]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(img)

        enc = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        if self.train:
            enc["labels"] = enc["input_ids"].clone()

        return enc


def train_baseline(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    train_csv="data/train.csv",
    data_dir="data",
    output_dir="checkpoints/baseline",
    image_size=384,
    n_samples=None,
    batch_size=1,
    grad_accum=4,
    learning_rate=1e-4,
    num_epochs=1,
    lora_r=8,
    lora_alpha=16,
    seed=42
):
    """
    Baseline 스타일 학습

    Args:
        model_id: Model ID (3B or 7B)
        train_csv: Training CSV path
        data_dir: Base directory for images
        output_dir: Output directory
        image_size: Image size (384 or 448)
        n_samples: Number of samples (None for all)
        batch_size: Batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        num_epochs: Number of epochs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        seed: Random seed
    """
    # Seed 설정
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("Baseline Training")
    print(f"{'='*60}\n")
    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Image size: {image_size}x{image_size}")

    # 데이터 로드
    print(f"\nLoading data from {train_csv}...")
    train_df = pd.read_csv(train_csv)

    if n_samples:
        train_df = train_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
        print(f"Using {n_samples} samples")
    else:
        print(f"Using all {len(train_df)} samples")

    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 프로세서
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=image_size*image_size,
        max_pixels=image_size*image_size,
        trust_remote_code=True,
    )

    # 모델
    print(f"Loading model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 준비
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 데이터셋 분할
    split = int(len(train_df) * 0.9)
    train_subset = train_df.iloc[:split]
    valid_subset = train_df.iloc[split:]

    print(f"\nTrain: {len(train_subset)} samples")
    print(f"Valid: {len(valid_subset)} samples")

    # 데이터셋 생성
    train_ds = VQAMCDataset(train_subset, processor, data_dir, train=True)
    valid_ds = VQAMCDataset(valid_subset, processor, data_dir, train=True)

    # 데이터로더
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollator(processor, True),
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(processor, True),
        num_workers=0
    )

    # 옵티마이저, 스케줄러
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * math.ceil(len(train_loader) / grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(num_training_steps * 0.03),
        num_training_steps
    )

    # 스케일러
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # 학습 루프
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}\n")

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", unit="batch")

        for step, batch in enumerate(progress_bar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            running += loss.item()

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                avg_loss = running / grad_accum
                progress_bar.set_postfix({"loss": f"{avg_loss:.3f}"})
                running = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            for vb in tqdm(valid_loader, desc=f"Epoch {epoch+1} [valid]", unit="batch"):
                vb = {k: v.to(device) for k, v in vb.items()}
                val_loss += model(**vb).loss.item()
                val_steps += 1

        print(f"[Epoch {epoch+1}] valid loss {val_loss/val_steps:.4f}")

    # 모델 저장
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}\n")
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Baseline-style VQA training')
    parser.add_argument('--model_id', default='Qwen/Qwen2.5-VL-3B-Instruct', help='Model ID')
    parser.add_argument('--train_csv', default='data/train.csv', help='Training CSV')
    parser.add_argument('--data_dir', default='data', help='Base data directory')
    parser.add_argument('--output_dir', default='checkpoints/baseline', help='Output directory')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--n_samples', type=int, help='Number of samples (None for all)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    train_baseline(
        model_id=args.model_id,
        train_csv=args.train_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
