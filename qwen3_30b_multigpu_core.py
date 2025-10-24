"""
Qwen3-VL-30B Multi-GPU 핵심 로직
T4 * 2 (32GB) 환경에서 안전하게 30B 모델 실행

주요 기능:
1. Multi-GPU Model Parallelism
2. 4-bit Quantization with QLoRA
3. Memory-efficient Training
4. Parallel Inference
"""

import torch
import torch.nn.functional as F
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import logging
import gc
from typing import Dict, List, Optional
from PIL import Image

logger = logging.getLogger('VQA_30B')

# ============================================================================
# 1. Multi-GPU 모델 로드 (핵심!)
# ============================================================================

def create_model_and_processor_multigpu(
    model_id: str,
    image_size: int = 384,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    max_memory_per_gpu: Dict[int, str] = None,
    use_gradient_checkpointing: bool = True,
    logger = None
):
    """
    Multi-GPU 환경에서 30B 모델 로드

    Args:
        model_id: 모델 ID (Qwen/Qwen2.5-VL-30B-A3B-Instruct)
        image_size: 이미지 크기
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA target modules
        max_memory_per_gpu: GPU당 최대 메모리 {0: "14GB", 1: "14GB"}
        use_gradient_checkpointing: Gradient checkpointing 사용 여부
        logger: 로거

    Returns:
        (model, processor)
    """
    if logger:
        logger.info("🔧 Multi-GPU 모델 로드 시작...")

    # GPU 확인
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 필요합니다!")

    gpu_count = torch.cuda.device_count()
    if logger:
        logger.info(f"   사용 가능 GPU: {gpu_count}개")

    # 기본 max_memory 설정
    if max_memory_per_gpu is None:
        if gpu_count >= 2:
            max_memory_per_gpu = {0: "14GB", 1: "14GB"}
        else:
            max_memory_per_gpu = {0: "14GB"}

    # 4-bit Quantization 설정 (필수!)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Double quantization
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4는 BF16 미지원
    )

    if logger:
        logger.info(f"   4-bit Quantization 설정 완료")
        logger.info(f"   Max memory per GPU: {max_memory_per_gpu}")

    # Processor 로드
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=image_size * image_size,
            max_pixels=image_size * image_size,
            trust_remote_code=True,
        )
        if logger:
            logger.info("✅ Processor 로드 완료")
    except Exception as e:
        if logger:
            logger.error(f"❌ Processor 로드 실패: {e}")
        raise

    # 모델 로드 with Multi-GPU
    try:
        if logger:
            logger.info("   Base model 로드 중...")

        # device_map="auto"로 자동 병렬화
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",  # 핵심! 자동으로 여러 GPU에 분산
            max_memory=max_memory_per_gpu,  # GPU당 최대 메모리
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # CPU 메모리 절약
        )

        if logger:
            logger.info("✅ Base model 로드 완료")
            # 모델이 어느 GPU에 배치되었는지 확인
            if hasattr(base_model, 'hf_device_map'):
                logger.info(f"   Device map: {base_model.hf_device_map}")

    except Exception as e:
        if logger:
            logger.error(f"❌ Model 로드 실패: {e}")
        raise

    # Gradient Checkpointing (메모리 절약)
    if use_gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if logger:
            logger.info("✅ Gradient checkpointing 활성화")

    # QLoRA 준비
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA Config
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # PEFT 모델 생성
    model = get_peft_model(base_model, lora_config)

    if logger:
        model.print_trainable_parameters()
        logger.info("✅ QLoRA 모델 생성 완료")

    # 메모리 상태 출력
    if logger:
        print_gpu_memory_status(logger)

    return model, processor


def print_gpu_memory_status(logger):
    """모든 GPU 메모리 상태 출력"""
    if not torch.cuda.is_available():
        return

    logger.info("="*60)
    logger.info("💾 GPU Memory Status")
    logger.info("="*60)

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        usage_pct = (allocated / total) * 100
        logger.info(
            f"GPU {i}: {allocated:.2f}GB / {total:.1f}GB ({usage_pct:.1f}%) | "
            f"Reserved: {reserved:.2f}GB"
        )

    logger.info("="*60)


def clear_gpu_memory():
    """모든 GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# ============================================================================
# 2. Memory-Efficient Training Loop
# ============================================================================

def train_one_epoch_memory_efficient(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    grad_accum_steps: int,
    max_grad_norm: float,
    device,
    logger=None
):
    """
    메모리 효율적인 학습 루프

    Args:
        model: 모델
        train_loader: 데이터 로더
        optimizer: 옵티마이저
        scheduler: 스케줄러
        scaler: AMP scaler
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Max gradient norm
        device: 디바이스
        logger: 로거

    Returns:
        average_loss
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # 데이터를 device로 이동 (multi-GPU의 경우 자동 처리)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward with AMP
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 메모리 절약

            scheduler.step()
            steps += 1

            # 주기적으로 메모리 정리
            if steps % 50 == 0:
                clear_gpu_memory()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# ============================================================================
# 3. Parallel Inference (Multi-GPU)
# ============================================================================

def infer_parallel(
    model,
    processor,
    test_df,
    data_dir: str,
    img_col: str = 'path',
    system_instruct: str = "",
    logger=None
):
    """
    Multi-GPU 병렬 추론

    Args:
        model: 모델 (이미 Multi-GPU에 분산됨)
        processor: Processor
        test_df: Test 데이터프레임
        data_dir: 데이터 디렉토리
        img_col: 이미지 컬럼명
        system_instruct: System instruction
        logger: 로거

    Returns:
        predictions: List of predictions
    """
    model.eval()
    predictions = []

    # Choice token IDs 추출
    choice_tokens = get_choice_token_ids_robust(processor)

    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc="Inference"):
            row = test_df.iloc[idx]

            # 이미지 로드
            img_path = f"{data_dir}/{row[img_col]}"
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                img = Image.new('RGB', (384, 384), color='white')

            # 프롬프트 생성
            question_text = build_mc_prompt(
                row['question'], row['a'], row['b'], row['c'], row['d']
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruct}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question_text}
                ]}
            ]

            # Processor로 처리
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt")

            # 입력을 적절한 device로 이동 (첫 번째 GPU)
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

            # Forward
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Choice 확률 계산
            probs = extract_choice_probs_from_logits(logits, choice_tokens)
            pred = max(probs, key=probs.get)

            predictions.append(pred)

            # 주기적으로 메모리 정리
            if (idx + 1) % 100 == 0:
                clear_gpu_memory()

    return predictions


# ============================================================================
# 4. Helper Functions
# ============================================================================

def build_mc_prompt(question, a, b, c, d):
    """Multiple choice 프롬프트 생성"""
    return (
        f"{question}\\n"
        f"(a) {a}\\n(b) {b}\\n(c) {c}\\n(d) {d}\\n\\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


def get_choice_token_ids_robust(processor):
    """Choice token IDs 추출 (여러 변형 고려)"""
    choice_tokens = {}
    for choice in ['a', 'b', 'c', 'd']:
        variants = [choice, f" {choice}", f"{choice} ", choice.upper()]
        all_token_ids = set()
        for variant in variants:
            try:
                token_ids = processor.tokenizer.encode(variant, add_special_tokens=False)
                all_token_ids.update(token_ids)
            except:
                pass
        choice_tokens[choice] = list(all_token_ids)
    return choice_tokens


def extract_choice_probs_from_logits(logits, choice_tokens):
    """Logits에서 choice 확률 추출"""
    choice_logits = {}
    for choice, token_ids in choice_tokens.items():
        if len(token_ids) > 0:
            max_logit = max([logits[tid].item() for tid in token_ids])
            choice_logits[choice] = max_logit
        else:
            choice_logits[choice] = -float('inf')

    # Softmax
    logit_values = torch.tensor(list(choice_logits.values()))
    probs = F.softmax(logit_values, dim=0).numpy()

    return {choice: probs[i] for i, choice in enumerate(['a', 'b', 'c', 'd'])}


# ============================================================================
# 5. Accelerate 통합 (Advanced)
# ============================================================================

def setup_accelerator(
    gradient_accumulation_steps: int = 16,
    mixed_precision: str = "fp16",
    cpu_offload: bool = True
):
    """
    Accelerate 설정

    Args:
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Mixed precision ("fp16", "bf16", "no")
        cpu_offload: CPU offload 사용 여부

    Returns:
        Accelerator instance
    """
    from accelerate import Accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        cpu=cpu_offload,
    )

    return accelerator


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Qwen3-VL-30B Multi-GPU Core Functions")
    print("="*60)
    print()
    print("주요 함수:")
    print("1. create_model_and_processor_multigpu() - Multi-GPU 모델 로드")
    print("2. train_one_epoch_memory_efficient() - 메모리 효율적 학습")
    print("3. infer_parallel() - 병렬 추론")
    print("4. print_gpu_memory_status() - GPU 메모리 모니터링")
    print("5. clear_gpu_memory() - GPU 메모리 정리")
    print()
    print("="*60)
