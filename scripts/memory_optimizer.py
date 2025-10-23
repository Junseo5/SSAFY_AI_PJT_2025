"""
Kaggle VQA Challenge - GPU Memory Optimizer

This script provides GPU memory management and optimization for T4 GPUs:
- Memory monitoring
- Automatic configuration based on available memory
- Memory-efficient training arguments
"""

import torch
import gc
from typing import Dict
from transformers import TrainingArguments


class GPUMemoryManager:
    """GPU 메모리 관리자"""

    def __init__(self):
        self.peak_memory = 0.0

    @staticmethod
    def clear_cache():
        """
        GPU 캐시 정리

        Usage:
            GPUMemoryManager.clear_cache()
        """
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """
        메모리 사용 통계

        Returns:
            dict: {
                'allocated': float (GB),
                'reserved': float (GB),
                'max_allocated': float (GB)
            }
        """
        if not torch.cuda.is_available():
            return {}

        stats = {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }

        return stats

    @staticmethod
    def print_memory_stats():
        """메모리 통계 출력"""
        stats = GPUMemoryManager.get_memory_stats()

        if stats:
            print(f"\n{'─'*60}")
            print("GPU Memory Statistics")
            print(f"{'─'*60}")
            print(f"  Allocated:     {stats['allocated']:.2f} GB")
            print(f"  Reserved:      {stats['reserved']:.2f} GB")
            print(f"  Max Allocated: {stats['max_allocated']:.2f} GB")
            print(f"{'─'*60}\n")
        else:
            print("❌ CUDA not available")

    @staticmethod
    def get_available_memory() -> float:
        """
        사용 가능한 GPU 메모리 반환

        Returns:
            float: 사용 가능 메모리 (GB)
        """
        if not torch.cuda.is_available():
            return 0.0

        # 전체 메모리
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        # 현재 할당된 메모리
        allocated = torch.cuda.memory_allocated() / 1e9

        # 사용 가능 메모리
        available = total_memory - allocated

        return available

    @staticmethod
    def optimize_training_config(available_memory_gb: float = None) -> Dict:
        """
        사용 가능 메모리에 따른 최적 학습 설정

        Args:
            available_memory_gb: 사용 가능 메모리 (GB)
                                None이면 자동 감지

        Returns:
            dict: {
                'batch_size': int,
                'gradient_accumulation_steps': int,
                'use_gradient_checkpointing': bool,
                'compute_dtype': torch.dtype
            }

        ✅ CRITICAL: T4 호환성을 위해 torch.float16 사용
        """
        if available_memory_gb is None:
            if torch.cuda.is_available():
                available_memory_gb = GPUMemoryManager.get_available_memory()
            else:
                available_memory_gb = 15  # 기본값 (T4 single)

        print(f"📊 Optimizing for {available_memory_gb:.1f} GB available memory")

        # T4 GPU × 2 (30GB)
        if available_memory_gb >= 25:
            config = {
                'batch_size': 8,
                'gradient_accumulation_steps': 1,
                'use_gradient_checkpointing': False,
                'compute_dtype': torch.float16,  # ✅ T4 compatible
                'max_seq_length': 1024
            }
            print("  → Using T4×2 configuration (high performance)")

        # T4 GPU × 1.5 (20-25GB)
        elif available_memory_gb >= 18:
            config = {
                'batch_size': 6,
                'gradient_accumulation_steps': 1,
                'use_gradient_checkpointing': True,
                'compute_dtype': torch.float16,  # ✅ T4 compatible
                'max_seq_length': 768
            }
            print("  → Using T4×1.5 configuration (balanced)")

        # T4 GPU × 1 (15GB) - 기본
        elif available_memory_gb >= 12:
            config = {
                'batch_size': 4,
                'gradient_accumulation_steps': 2,
                'use_gradient_checkpointing': True,
                'compute_dtype': torch.float16,  # ✅ T4 compatible
                'max_seq_length': 512
            }
            print("  → Using T4×1 configuration (standard)")

        # 저메모리 (< 12GB)
        else:
            config = {
                'batch_size': 2,
                'gradient_accumulation_steps': 4,
                'use_gradient_checkpointing': True,
                'compute_dtype': torch.float16,  # ✅ T4 compatible
                'max_seq_length': 512
            }
            print("  → Using low-memory configuration")
            print("  ⚠️  Warning: Very limited memory. Consider closing other processes.")

        # 유효 배치 크기
        effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
        print(f"  → Effective batch size: {effective_batch_size}")

        return config

    def monitor_training(self, callback_interval: int = 100):
        """
        학습 중 메모리 모니터링 콜백 생성

        Args:
            callback_interval: 모니터링 간격 (steps)

        Returns:
            Callable: 콜백 함수

        Usage:
            manager = GPUMemoryManager()
            callback = manager.monitor_training(interval=100)

            # 학습 루프 내에서:
            callback(step=current_step)
        """
        def callback(step: int):
            if step % callback_interval == 0:
                stats = self.get_memory_stats()
                self.peak_memory = max(self.peak_memory, stats.get('allocated', 0))

                print(f"\nStep {step}:")
                print(f"  Memory: {stats['allocated']:.2f} GB / {stats['reserved']:.2f} GB")
                print(f"  Peak:   {self.peak_memory:.2f} GB")

                # 메모리 경고 (T4 15GB의 87%)
                if stats['allocated'] > 13:
                    print("  ⚠️  WARNING: Memory usage is high!")
                    self.clear_cache()
                    print("  → Cache cleared")

        return callback


def create_memory_efficient_training_args(
    output_dir: str = 'checkpoints',
    available_memory_gb: float = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-5
) -> TrainingArguments:
    """
    메모리 효율적인 학습 설정 생성

    Args:
        output_dir: 체크포인트 저장 디렉토리
        available_memory_gb: 사용 가능 메모리 (GB), None이면 자동
        num_epochs: 학습 에폭 수
        learning_rate: 학습률

    Returns:
        TrainingArguments: 메모리 최적화된 학습 설정

    ✅ CRITICAL: T4 호환성을 위해 fp16=True, bf16=False
    """
    # 메모리 기반 최적 설정
    manager = GPUMemoryManager()
    config = manager.optimize_training_config(available_memory_gb)

    print(f"\n{'='*60}")
    print("Creating Memory-Efficient Training Arguments")
    print(f"{'='*60}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,

        # Batch size (메모리 기반)
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],

        # Memory optimization
        gradient_checkpointing=config['use_gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False},  # 안정성

        # ✅ CRITICAL: T4 호환성
        fp16=True,              # ✅ Float16 사용
        bf16=False,             # ✅ BFloat16 미사용 (T4 unsupported)

        # Optimizer (메모리 효율)
        optim="paged_adamw_8bit",

        # Data loading (메모리 절약)
        dataloader_pin_memory=False,
        dataloader_num_workers=2,

        # Gradient clipping
        max_grad_norm=1.0,

        # Learning rate
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # Weight decay
        weight_decay=0.01,

        # Label smoothing
        label_smoothing_factor=0.05,

        # Logging
        logging_steps=10,
        logging_first_step=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=50,

        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # 최대 2개 체크포인트만 유지

        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",

        # WandB
        report_to="wandb",

        # Reproducibility
        seed=42,
        data_seed=42,

        # Disable unnecessary features
        push_to_hub=False,
    )

    print("Training Arguments:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Gradient checkpointing: {config['use_gradient_checkpointing']}")
    print(f"  Precision: FP16 (T4 compatible)")
    print(f"  Optimizer: paged_adamw_8bit")
    print(f"{'='*60}\n")

    return training_args


def main():
    """메인 실행 함수"""
    print("="*60)
    print("GPU Memory Optimizer")
    print("="*60 + "\n")

    # GPU 정보
    if torch.cuda.is_available():
        print(f"CUDA Available: ✓")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: ✗")
        print("Running on CPU")

    # 메모리 통계
    manager = GPUMemoryManager()
    manager.print_memory_stats()

    # 사용 가능 메모리
    available = manager.get_available_memory()
    print(f"Available Memory: {available:.2f} GB\n")

    # 최적 설정
    config = manager.optimize_training_config()

    print("\nOptimal Configuration:")
    for key, value in config.items():
        print(f"  {key:30s}: {value}")

    # 학습 설정 예시
    print("\n" + "="*60)
    print("Example: Creating Training Arguments")
    print("="*60 + "\n")

    training_args = create_memory_efficient_training_args(
        output_dir='checkpoints/example',
        available_memory_gb=15,  # T4 single
        num_epochs=3,
        learning_rate=2e-5
    )

    print("✅ Training arguments created successfully")


if __name__ == "__main__":
    main()
