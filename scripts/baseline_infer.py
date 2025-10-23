"""
Kaggle VQA Challenge - Baseline-Style Inference Script

This is a simplified inference script following the baseline notebook structure.
Perfect for quick testing and generating submissions.

Features:
- Simple and straightforward
- Direct path column usage
- Compatible with AutoModelForVision2Seq
- Fast prediction
"""

import os, re
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
import argparse
from pathlib import Path

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


def predict_single(model, processor, image_path, question, a, b, c, d, device="cuda"):
    """
    단일 샘플 예측

    Args:
        model: Fine-tuned model
        processor: AutoProcessor
        image_path: 이미지 경로
        question: 질문
        a, b, c, d: 보기
        device: 디바이스

    Returns:
        str: 예측된 답 (a/b/c/d)
    """
    # 이미지 로드
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"⚠️  Warning: Image not found: {image_path}, using blank image")
        img = Image.new('RGB', (384, 384), color='white')

    # 프롬프트 생성
    user_text = build_mc_prompt(question, a, b, c, d)

    # 메시지 구성
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_text}
        ]}
    ]

    # 텍스트 생성
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 인코딩
    inputs = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0
        )

    # 디코딩
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)

    # 답변 파싱
    answer = parse_answer(generated_text)

    return answer


def parse_answer(text):
    """
    생성된 텍스트에서 답 추출

    Args:
        text: 생성된 텍스트

    Returns:
        str: a/b/c/d 중 하나
    """
    text = text.lower()

    # assistant 메시지 이후 추출
    if "assistant" in text:
        text = text.split("assistant")[-1]

    # a, b, c, d 찾기
    matches = re.findall(r'\b([abcd])\b', text)

    if matches:
        return matches[0]
    else:
        # Fallback: 텍스트 전체에서 첫 번째 a/b/c/d
        for char in text:
            if char in 'abcd':
                return char
        return 'a'  # 최종 fallback


def infer_baseline(
    model_path="checkpoints/baseline",
    test_csv="data/test.csv",
    data_dir="data",
    output_csv="outputs/submission_baseline.csv",
    image_size=384,
    device="cuda"
):
    """
    Baseline 스타일 추론

    Args:
        model_path: Fine-tuned 모델 경로
        test_csv: 테스트 CSV 경로
        data_dir: 데이터 디렉토리
        output_csv: 출력 CSV 경로
        image_size: 이미지 크기
        device: 디바이스
    """
    print(f"\n{'='*60}")
    print("Baseline Inference")
    print(f"{'='*60}\n")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}x{image_size}")

    # 프로세서 로드
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=image_size*image_size,
        max_pixels=image_size*image_size,
        trust_remote_code=True,
    )

    # 모델 로드
    print(f"Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()

    print(f"✓ Model loaded\n")

    # 테스트 데이터 로드
    print(f"Loading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    print(f"Total test samples: {len(test_df)}\n")

    # 예측
    results = []

    print(f"{'─'*60}")
    print("Running predictions...")
    print(f"{'─'*60}\n")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        # 이미지 경로 처리
        if 'path' in row:
            img_path = os.path.join(data_dir, row["path"])
        elif 'image' in row:
            img_path = os.path.join(data_dir, row["image"])
        else:
            raise ValueError("No 'path' or 'image' column found")

        # 예측
        try:
            answer = predict_single(
                model, processor, img_path,
                str(row["question"]),
                str(row["a"]), str(row["b"]),
                str(row["c"]), str(row["d"]),
                device
            )
        except Exception as e:
            print(f"⚠️  Error predicting {row['id']}: {e}")
            answer = 'a'  # Fallback

        results.append({
            'id': row['id'],
            'answer': answer
        })

    # 제출 파일 생성
    submission_df = pd.DataFrame(results)

    # 정렬
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    # 저장
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print("Inference Complete")
    print(f"{'='*60}\n")
    print(f"✓ Submission saved to {output_csv}")
    print(f"  Total predictions: {len(submission_df)}")

    # 답변 분포
    print("\nAnswer distribution:")
    for ans, count in submission_df['answer'].value_counts().sort_index().items():
        percentage = count / len(submission_df) * 100
        print(f"  {ans}: {count:5d} ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Baseline-style VQA inference')
    parser.add_argument('--model_path', default='checkpoints/baseline', help='Model checkpoint path')
    parser.add_argument('--test_csv', default='data/test.csv', help='Test CSV')
    parser.add_argument('--data_dir', default='data', help='Base data directory')
    parser.add_argument('--output_csv', default='outputs/submission_baseline.csv', help='Output CSV path')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--device', default='cuda', help='Device')

    args = parser.parse_args()

    infer_baseline(
        model_path=args.model_path,
        test_csv=args.test_csv,
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        image_size=args.image_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
