"""
Kaggle VQA Challenge - Stratified Cross-Validation

This script creates stratified K-fold splits that maintain:
- Question type distribution
- Answer distribution
- Balanced folds for better model evaluation
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import re
from typing import Tuple
from pathlib import Path


class VQAStratifiedSplitter:
    """질문 유형 비율 유지 K-Fold 분할기"""

    def __init__(self, n_folds: int = 3, seed: int = 42):
        """
        Args:
            n_folds: Fold 개수
            seed: Random seed (재현성)
        """
        self.n_folds = n_folds
        self.seed = seed
        self.type_patterns = {
            'counting': r'몇|개수|수|how many|count',
            'color': r'색|색깔|color|무슨색',
            'ocr': r'글자|문자|숫자|번호|읽|text|number|write|written',
            'yesno': r'인가|입니까|\?$|있는가|맞는가|yes|no',
            'location': r'어디|위치|where|장소|place',
            'attribute': r'무엇|what|어떤|kind|which'
        }

    def _classify_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        질문 유형 자동 분류

        Args:
            df: 입력 DataFrame

        Returns:
            DataFrame: 'question_type' 컬럼 추가
        """
        def classify(question: str) -> str:
            """단일 질문 분류"""
            if pd.isna(question):
                return 'general'

            question_lower = question.lower()

            for qtype, pattern in self.type_patterns.items():
                if re.search(pattern, question_lower, re.I):
                    return qtype
            return 'general'

        print("📊 Classifying question types...")
        df['question_type'] = df['question'].apply(classify)

        # 분포 출력
        type_counts = df['question_type'].value_counts()
        print("\nQuestion Type Distribution:")
        for qtype, count in type_counts.items():
            percentage = count / len(df) * 100
            print(f"  {qtype:12s}: {count:4d} ({percentage:5.1f}%)")

        return df

    def create_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stratified K-Fold 생성

        Args:
            df: train.csv 데이터프레임

        Returns:
            pd.DataFrame: 'fold' 컬럼 추가된 데이터프레임
        """
        print("\n" + "="*60)
        print(f"Creating {self.n_folds}-Fold Stratified CV Splits")
        print("="*60 + "\n")

        # 1. 질문 유형 분류
        df = self._classify_questions(df)

        # 2. Stratify 레이블 생성 (question_type + answer)
        # 이렇게 하면 질문 유형과 정답 분포가 모두 유지됨
        print("\n🔄 Creating stratification labels...")
        df['stratify_label'] = df['question_type'] + '_' + df['answer'].astype(str)

        # 레이블 분포 확인
        label_counts = df['stratify_label'].value_counts()
        print(f"✓ Created {len(label_counts)} unique stratification labels")

        # 희귀 레이블 처리 (너무 적은 경우 일반 레이블로 변경)
        min_samples_per_fold = 2
        rare_labels = label_counts[label_counts < self.n_folds * min_samples_per_fold].index

        if len(rare_labels) > 0:
            print(f"⚠️  Found {len(rare_labels)} rare labels (< {self.n_folds * min_samples_per_fold} samples)")
            print(f"   Merging rare labels into 'general' type...")

            for label in rare_labels:
                # 희귀 레이블을 general_answer 형태로 변경
                answer = label.split('_')[-1]
                df.loc[df['stratify_label'] == label, 'stratify_label'] = f'general_{answer}'

        # 3. Stratified K-Fold 실행
        print(f"\n🔀 Performing {self.n_folds}-fold stratified split (seed={self.seed})...")

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
            print(f"  Fold {fold_idx}: {len(train_idx)} train / {len(val_idx)} val")

        # 4. 분포 확인
        self._print_fold_distribution(df)

        # 5. Stratify 레이블 제거 (임시 컬럼)
        df = df.drop(columns=['stratify_label'])

        print("\n✓ Fold creation complete!")
        return df

    def _print_fold_distribution(self, df: pd.DataFrame):
        """
        Fold별 분포 출력

        Args:
            df: Fold 정보가 포함된 DataFrame
        """
        print("\n" + "="*60)
        print("📊 Fold Distribution Analysis")
        print("="*60)

        for fold in range(self.n_folds):
            fold_df = df[df['fold'] == fold]

            print(f"\n{'─'*60}")
            print(f"Fold {fold} ({len(fold_df)} samples / {len(fold_df)/len(df)*100:.1f}%)")
            print(f"{'─'*60}")

            # 질문 유형 분포
            print("\n  Question Type Distribution:")
            type_dist = fold_df['question_type'].value_counts()
            for qtype in sorted(df['question_type'].unique()):
                count = type_dist.get(qtype, 0)
                total_count = (df['question_type'] == qtype).sum()
                pct_in_fold = count / len(fold_df) * 100 if len(fold_df) > 0 else 0
                pct_of_type = count / total_count * 100 if total_count > 0 else 0
                print(f"    {qtype:12s}: {count:4d} ({pct_in_fold:5.1f}% in fold, {pct_of_type:5.1f}% of type)")

            # 정답 분포
            print("\n  Answer Distribution:")
            answer_dist = fold_df['answer'].value_counts().sort_index()
            for ans in ['a', 'b', 'c', 'd']:
                count = answer_dist.get(ans, 0)
                total_count = (df['answer'] == ans).sum()
                pct_in_fold = count / len(fold_df) * 100 if len(fold_df) > 0 else 0
                pct_of_ans = count / total_count * 100 if total_count > 0 else 0
                print(f"    {ans}: {count:4d} ({pct_in_fold:5.1f}% in fold, {pct_of_ans:5.1f}% of answer)")

        # 전체 균형 체크
        print("\n" + "="*60)
        print("Balance Check")
        print("="*60)

        # Fold별 크기 균형
        fold_sizes = [len(df[df['fold'] == i]) for i in range(self.n_folds)]
        expected_size = len(df) / self.n_folds
        max_deviation = max(abs(size - expected_size) for size in fold_sizes)

        print(f"\nFold size balance:")
        print(f"  Expected: {expected_size:.0f}")
        print(f"  Actual:   {fold_sizes}")
        print(f"  Max deviation: {max_deviation:.0f} ({max_deviation/expected_size*100:.1f}%)")

        if max_deviation / expected_size < 0.05:
            print("  ✓ Fold sizes are well balanced")
        else:
            print("  ⚠️  Fold sizes may be imbalanced")

        # 질문 유형 균형
        print(f"\nQuestion type balance:")
        for qtype in df['question_type'].unique():
            type_df = df[df['question_type'] == qtype]
            fold_counts = [len(type_df[type_df['fold'] == i]) for i in range(self.n_folds)]
            expected = len(type_df) / self.n_folds
            max_dev = max(abs(count - expected) for count in fold_counts) if expected > 0 else 0
            dev_pct = max_dev / expected * 100 if expected > 0 else 0

            print(f"  {qtype:12s}: {fold_counts} (max dev: {dev_pct:.1f}%)")

    def get_fold_data(self, df: pd.DataFrame, fold: int, mode: str = 'train') -> pd.DataFrame:
        """
        특정 fold의 train 또는 val 데이터 추출

        Args:
            df: Fold 정보가 포함된 DataFrame
            fold: Fold 번호 (0부터 시작)
            mode: 'train' 또는 'val'

        Returns:
            DataFrame: 추출된 데이터
        """
        if mode == 'train':
            return df[df['fold'] != fold].reset_index(drop=True)
        elif mode == 'val':
            return df[df['fold'] == fold].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")

    def save_folds(self, df: pd.DataFrame, output_path: str = 'data/train_with_folds.csv'):
        """
        Fold 정보가 포함된 DataFrame 저장

        Args:
            df: Fold 정보가 포함된 DataFrame
            output_path: 출력 파일 경로
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved fold information to {output_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Create stratified CV folds for VQA dataset')
    parser.add_argument('--input_csv', default='data/train.csv', help='Input train CSV')
    parser.add_argument('--output_csv', default='data/train_with_folds.csv', help='Output CSV with fold column')
    parser.add_argument('--n_folds', type=int, default=3, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("="*60)
    print("VQA Stratified Cross-Validation")
    print("="*60 + "\n")

    # 데이터 로드
    print(f"📁 Loading data from {args.input_csv}...")
    train_df = pd.read_csv(args.input_csv)
    print(f"✓ Loaded {len(train_df)} samples\n")

    # Splitter 생성
    splitter = VQAStratifiedSplitter(n_folds=args.n_folds, seed=args.seed)

    # Fold 생성
    train_df_with_folds = splitter.create_folds(train_df)

    # 저장
    splitter.save_folds(train_df_with_folds, args.output_csv)

    # 예시: Fold 0의 train/val 데이터 추출
    print("\n" + "="*60)
    print("Example: Fold 0 Train/Val Split")
    print("="*60)

    fold0_train = splitter.get_fold_data(train_df_with_folds, fold=0, mode='train')
    fold0_val = splitter.get_fold_data(train_df_with_folds, fold=0, mode='val')

    print(f"\nFold 0:")
    print(f"  Train: {len(fold0_train)} samples")
    print(f"  Val:   {len(fold0_val)} samples")

    print("\n✅ Cross-validation splits created successfully!")


if __name__ == "__main__":
    main()
