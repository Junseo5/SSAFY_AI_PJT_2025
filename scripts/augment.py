"""
Kaggle VQA Challenge - Data Augmentation

This script handles data augmentation including:
- Choice shuffling with answer label update
- Korean question paraphrasing
- Image augmentation (excluding OCR questions)
"""

import random
from PIL import Image, ImageEnhance
import re
import os
from pathlib import Path
from typing import Dict, List
import numpy as np


class VQAAugmenter:
    """VQA 데이터 증강 클래스"""

    def __init__(self, config: Dict = None):
        """
        Args:
            config: 증강 설정
                {
                    'shuffle_choices': True,
                    'paraphrase_question': True,
                    'image_aug': True,
                    'ocr_question_types': ['ocr']
                }
        """
        self.config = config or {
            'shuffle_choices': True,
            'paraphrase_question': True,
            'image_aug': True,
            'ocr_question_types': ['ocr']
        }

        # 한국어 paraphrase 규칙
        self.paraphrase_rules = {
            r'몇\s*개': ['개수는', '몇 개가', '수량은'],
            r'무슨\s*색': ['어떤 색', '색깔은', '무슨 색깔'],
            r'있습니까': ['있나요', '있는가', '존재하나요'],
            r'어디': ['어느 곳', '어느 장소', '어디에'],
            r'무엇': ['뭐', '무엇이', '어떤 것'],
        }

    def augment_sample(
        self,
        image_path: str,
        question: str,
        choices: Dict[str, str],
        answer: str,
        question_type: str = 'general'
    ) -> List[Dict]:
        """
        단일 샘플 증강

        Args:
            image_path: 이미지 경로
            question: 질문 텍스트
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}
            answer: 정답 ('a', 'b', 'c', 'd' 중 하나)
            question_type: 질문 유형 (OCR 판별용)

        Returns:
            list: 증강된 샘플들
        """
        augmented = []

        # 1. 원본 샘플 추가
        augmented.append({
            'image': image_path,
            'question': question,
            'choices': choices.copy(),
            'answer': answer,
            'question_type': question_type,
            'augmentation': 'original'
        })

        # 2. 보기 순서 셔플
        if self.config.get('shuffle_choices', True):
            shuffled = self._shuffle_choices(choices, answer)
            augmented.append({
                'image': image_path,
                'question': question,
                'choices': shuffled['choices'],
                'answer': shuffled['answer'],
                'question_type': question_type,
                'augmentation': 'shuffle_choices'
            })

        # 3. 질문 paraphrase (한국어만)
        if self.config.get('paraphrase_question', True):
            para_q = self._paraphrase_korean(question)
            if para_q != question:  # 변형된 경우만
                augmented.append({
                    'image': image_path,
                    'question': para_q,
                    'choices': choices.copy(),
                    'answer': answer,
                    'question_type': question_type,
                    'augmentation': 'paraphrase_question'
                })

        # 4. 이미지 증강 (✅ CRITICAL: OCR 문제 제외)
        if self.config.get('image_aug', True):
            is_ocr = question_type in self.config.get('ocr_question_types', ['ocr'])

            if not is_ocr:
                try:
                    aug_img = self._augment_image(image_path)
                    augmented.append({
                        'image': aug_img,
                        'question': question,
                        'choices': choices.copy(),
                        'answer': answer,
                        'question_type': question_type,
                        'augmentation': 'image_aug'
                    })
                except Exception as e:
                    print(f"⚠️  Image augmentation failed for {image_path}: {e}")
            else:
                # OCR 문제는 이미지 증강 제외
                pass

        return augmented

    def _shuffle_choices(self, choices: Dict[str, str], answer: str) -> Dict:
        """
        보기 순서 무작위화 + 정답 라벨 업데이트

        Args:
            choices: {'a': '...', 'b': '...', 'c': '...', 'd': '...'}
            answer: 정답 ('a', 'b', 'c', 'd')

        Returns:
            dict: {
                'choices': {'a': '...', ...},
                'answer': 'b'  # 업데이트된 정답
            }
        """
        # 인덱스 매핑
        mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        reverse_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

        # 보기 리스트
        choice_list = [choices['a'], choices['b'], choices['c'], choices['d']]
        correct_idx = mapping[answer.lower()]

        # 셔플 (인덱스와 함께)
        paired = list(zip(choice_list, range(4)))
        random.shuffle(paired)
        shuffled_choices, original_indices = zip(*paired)

        # 새 정답 찾기
        new_answer_idx = original_indices.index(correct_idx)
        new_answer = reverse_mapping[new_answer_idx]

        return {
            'choices': {
                'a': shuffled_choices[0],
                'b': shuffled_choices[1],
                'c': shuffled_choices[2],
                'd': shuffled_choices[3]
            },
            'answer': new_answer
        }

    def _paraphrase_korean(self, question: str) -> str:
        """
        한국어 질문 변형

        Args:
            question: 원본 질문

        Returns:
            str: 변형된 질문 (변형 불가능하면 원본 반환)

        Examples:
            "몇 개" → "개수는", "몇 개가"
            "무슨 색" → "어떤 색", "색깔은"
        """
        # 한글이 없으면 변형 불가
        if not re.search(r'[가-힣]', question):
            return question

        modified = question

        for pattern, alternatives in self.paraphrase_rules.items():
            if re.search(pattern, modified):
                alt = random.choice(alternatives)
                modified = re.sub(pattern, alt, modified, count=1)
                # 하나만 변형
                break

        return modified

    def _augment_image(self, image_path: str, output_suffix: str = '_aug') -> str:
        """
        경량 이미지 증강 (OCR 문제 제외)

        Transformations:
            - Brightness: 0.9~1.1
            - Contrast: 0.95~1.05
            ❌ 제외: Flip, Rotation (OCR 깨짐, 객체 방향 변경)

        Args:
            image_path: 원본 이미지 경로
            output_suffix: 출력 파일 suffix

        Returns:
            str: 증강된 이미지 경로
        """
        # 이미지 로드
        img = Image.open(image_path)

        # 1. 밝기 조정 (0.9~1.1)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

        # 2. 대비 조정 (0.95~1.05)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

        # 3. 선명도 조정 (0.95~1.05) - 선택적
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.95, 1.05))

        # 저장 경로 생성 (확장자 보존)
        path_obj = Path(image_path)
        base = path_obj.stem
        ext = path_obj.suffix
        parent = path_obj.parent

        # 증강 이미지 저장 경로
        aug_path = parent / f"{base}{output_suffix}{ext}"

        # 저장 (고품질)
        img.save(aug_path, quality=95)

        return str(aug_path)

    def augment_batch(
        self,
        df,
        image_dir: str,
        output_csv: str = None
    ):
        """
        DataFrame 일괄 증강

        Args:
            df: 원본 DataFrame (train.csv)
            image_dir: 이미지 디렉토리
            output_csv: 증강된 데이터 저장 경로 (선택)

        Returns:
            DataFrame: 증강된 DataFrame
        """
        import pandas as pd
        from tqdm import tqdm

        print(f"🔄 Augmenting {len(df)} samples...")

        augmented_rows = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            # 이미지 경로
            if 'image' in row:
                image_path = os.path.join(image_dir, row['image'])
            else:
                # 이미지 경로가 없으면 스킵
                continue

            # 보기 구성
            choices = {
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'd': row['d']
            }

            # 질문 유형 (없으면 general)
            question_type = row.get('question_type', 'general')

            # 증강 실행
            augmented_samples = self.augment_sample(
                image_path=image_path,
                question=row['question'],
                choices=choices,
                answer=row['answer'],
                question_type=question_type
            )

            # DataFrame 행으로 변환
            for aug_sample in augmented_samples:
                new_row = row.copy()
                new_row['image'] = Path(aug_sample['image']).name
                new_row['question'] = aug_sample['question']
                new_row['a'] = aug_sample['choices']['a']
                new_row['b'] = aug_sample['choices']['b']
                new_row['c'] = aug_sample['choices']['c']
                new_row['d'] = aug_sample['choices']['d']
                new_row['answer'] = aug_sample['answer']
                new_row['augmentation'] = aug_sample['augmentation']

                augmented_rows.append(new_row)

        # DataFrame 생성
        augmented_df = pd.DataFrame(augmented_rows)

        print(f"✓ Augmented to {len(augmented_df)} samples (from {len(df)})")
        print(f"  Augmentation factor: {len(augmented_df) / len(df):.1f}x")

        # 증강 방법별 분포
        print("\nAugmentation Method Distribution:")
        aug_counts = augmented_df['augmentation'].value_counts()
        for method, count in aug_counts.items():
            print(f"  {method:20s}: {count:5d} ({count/len(augmented_df)*100:5.1f}%)")

        # 저장
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            augmented_df.to_csv(output_csv, index=False)
            print(f"\n✓ Saved augmented data to {output_csv}")

        return augmented_df


def main():
    """메인 실행 함수"""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Augment VQA dataset')
    parser.add_argument('--input_csv', default='data/train_with_folds.csv', help='Input CSV')
    parser.add_argument('--image_dir', default='data/images', help='Image directory')
    parser.add_argument('--output_csv', default='data/train_augmented.csv', help='Output CSV')
    parser.add_argument('--shuffle_choices', action='store_true', default=True, help='Shuffle choices')
    parser.add_argument('--paraphrase', action='store_true', default=True, help='Paraphrase questions')
    parser.add_argument('--image_aug', action='store_true', default=True, help='Augment images')

    args = parser.parse_args()

    print("="*60)
    print("VQA Data Augmentation")
    print("="*60 + "\n")

    # 설정
    config = {
        'shuffle_choices': args.shuffle_choices,
        'paraphrase_question': args.paraphrase,
        'image_aug': args.image_aug,
        'ocr_question_types': ['ocr']
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print()

    # 데이터 로드
    print(f"📁 Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"✓ Loaded {len(df)} samples\n")

    # 증강기 생성
    augmenter = VQAAugmenter(config=config)

    # 증강 실행
    augmented_df = augmenter.augment_batch(
        df=df,
        image_dir=args.image_dir,
        output_csv=args.output_csv
    )

    print("\n✅ Augmentation complete!")


if __name__ == "__main__":
    main()
