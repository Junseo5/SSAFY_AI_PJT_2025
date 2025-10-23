"""
Kaggle VQA Challenge - Forced-Choice Inference

✅ CRITICAL FIXES APPLIED:
1. Qwen2_5_VLForConditionalGeneration (correct class)
2. AutoProcessor (correct processor)
3. torch.float16 (T4 compatible)
4. attn_implementation="sdpa" (no FlashAttention)
5. process_vision_info (required)
6. apply_chat_template (standard method)
7. Unified prompt templates
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import pandas as pd
from tqdm import tqdm
import re
import os
from pathlib import Path
from typing import Dict


class ForcedChoicePredictor:
    """
    Forced-choice VQA 예측기

    ✅ CRITICAL: 프롬프트 템플릿 통일, Qwen2.5-VL 호환
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        prompt_manager=None
    ):
        """
        Args:
            model_path: 체크포인트 경로
            device: GPU 디바이스
            prompt_manager: PromptManager 인스턴스 (선택)
        """
        self.device = device
        self.model_path = model_path

        print(f"\n{'='*60}")
        print("Loading Model for Inference")
        print(f"{'='*60}\n")
        print(f"Model path: {model_path}")
        print(f"Device: {device}")

        # ✅ CRITICAL: 모델 로드
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # ✅ FP16
            attn_implementation="sdpa"   # ✅ SDPA
        )
        self.model.eval()

        print("✓ Model loaded")

        # ✅ CRITICAL: AutoProcessor 로드
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=768*28*28
        )

        print("✓ Processor loaded")

        # ✅ PromptManager 초기화
        if prompt_manager is None:
            import sys
            sys.path.append('.')
            from scripts.prompt_manager import PromptManager
            self.prompt_manager = PromptManager()
        else:
            self.prompt_manager = prompt_manager

        print("✓ Prompt manager initialized\n")

    def predict(
        self,
        image_path: str,
        question: str,
        choices: Dict[str, str],
        question_type: str = 'general'
    ) -> Dict:
        """
        단일 샘플 예측

        ✅ CRITICAL: 프롬프트 템플릿 통일, apply_chat_template 사용

        Args:
            image_path: 이미지 경로
            question: 질문
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}
            question_type: 질문 유형

        Returns:
            dict: {
                'prediction': 'a' | 'b' | 'c' | 'd',
                'confidence': float,
                'scores': dict
            }
        """
        # ✅ CRITICAL: PromptManager로 메시지 생성
        messages = self.prompt_manager.build_messages(
            image_path, question_type, question, choices
        )

        # ✅ CRITICAL: apply_chat_template 사용
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ✅ 추론 시에는 True
        )

        # ✅ CRITICAL: process_vision_info 사용
        images, videos = process_vision_info(messages)

        # 인코딩
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # 추론 (1-token generation)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                return_dict_in_generate=True,
                output_scores=True
            )

        # ✅ Forced-choice 로직 (로짓 기반)
        if len(outputs.scores) > 0:
            logits = outputs.scores[0][0]  # (vocab_size,)
            logp = torch.log_softmax(logits, dim=-1)

            # a/b/c/d 토큰 ID 수집
            def get_token_ids(char):
                """다양한 형태의 토큰 ID 수집"""
                variants = [char, " " + char, "\n" + char, char + " "]
                token_ids = set()
                for variant in variants:
                    ids = self.processor.tokenizer.encode(
                        variant,
                        add_special_tokens=False
                    )
                    token_ids.update(ids)
                return list(token_ids)

            # 각 선택지의 로그 확률
            scores = {}
            for c in ['a', 'b', 'c', 'd']:
                token_ids = get_token_ids(c)
                if token_ids:
                    scores[c] = torch.logsumexp(logp[token_ids], dim=0).item()
                else:
                    scores[c] = -float('inf')

            # 예측
            prediction = max(scores, key=scores.get)

            # Confidence (margin)
            sorted_scores = sorted(scores.values(), reverse=True)
            confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        else:
            # Fallback
            generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
            prediction = self._parse_answer(generated_text)
            scores = {}
            confidence = 0.0

        return {
            'prediction': prediction,
            'confidence': confidence,
            'scores': scores
        }

    def _parse_answer(self, text: str) -> str:
        """
        답변 파싱 (Fallback)

        Args:
            text: 생성된 텍스트

        Returns:
            str: 'a', 'b', 'c', 'd' 중 하나
        """
        text = text.lower()

        # a, b, c, d 찾기
        matches = re.findall(r'\b[abcd]\b', text)

        if matches:
            return matches[0]
        else:
            return 'a'

    def predict_batch(
        self,
        test_csv: str,
        image_dir: str,
        output_csv: str
    ):
        """
        배치 예측

        ✅ CRITICAL: 질문 유형 자동 판별

        Args:
            test_csv: 테스트 CSV 경로
            image_dir: 이미지 디렉토리
            output_csv: 출력 CSV 경로
        """
        print(f"\n{'='*60}")
        print("Batch Prediction")
        print(f"{'='*60}\n")

        test_df = pd.read_csv(test_csv)
        print(f"Total test samples: {len(test_df)}")

        # 질문 유형 분류
        import sys
        sys.path.append('.')
        from scripts.stratified_cv import VQAStratifiedSplitter

        splitter = VQAStratifiedSplitter()
        test_df = splitter._classify_questions(test_df)

        print("\nQuestion type distribution:")
        for qtype, count in test_df['question_type'].value_counts().items():
            print(f"  {qtype:12s}: {count:4d}")

        results = []

        print(f"\n{'─'*60}")
        print("Running predictions...")
        print(f"{'─'*60}\n")

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            image_path = os.path.join(image_dir, row['image'])

            # 이미지 존재 확인
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image not found: {image_path}, using fallback")
                results.append({
                    'id': row['id'],
                    'answer': 'a'
                })
                continue

            choices = {
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'd': row['d']
            }

            question_type = row.get('question_type', 'general')

            try:
                result = self.predict(
                    image_path,
                    row['question'],
                    choices,
                    question_type
                )

                results.append({
                    'id': row['id'],
                    'answer': result['prediction']
                })
            except Exception as e:
                print(f"⚠️ Error predicting {row['id']}: {e}")
                results.append({
                    'id': row['id'],
                    'answer': 'a'
                })

        # 제출 파일 생성
        submission_df = pd.DataFrame(results)

        # 정렬 (ID 순서)
        submission_df = submission_df.sort_values('id').reset_index(drop=True)

        # 저장
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_csv, index=False)

        print(f"\n{'='*60}")
        print("Prediction Complete")
        print(f"{'='*60}\n")
        print(f"✓ Submission saved to {output_csv}")
        print(f"  Total predictions: {len(submission_df)}")

        # 답변 분포
        print("\nAnswer distribution:")
        for ans, count in submission_df['answer'].value_counts().sort_index().items():
            print(f"  {ans}: {count:4d} ({count/len(submission_df)*100:5.1f}%)")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='VQA Forced-Choice Inference')
    parser.add_argument('--model_path', required=True, help='Model checkpoint path')
    parser.add_argument('--test_csv', default='data/test.csv', help='Test CSV path')
    parser.add_argument('--image_dir', default='data/images', help='Image directory')
    parser.add_argument('--output_csv', default='outputs/submission.csv', help='Output CSV path')
    parser.add_argument('--device', default='cuda:0', help='Device')

    args = parser.parse_args()

    # 예측기 생성
    predictor = ForcedChoicePredictor(args.model_path, args.device)

    # 배치 예측
    predictor.predict_batch(args.test_csv, args.image_dir, args.output_csv)

    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
