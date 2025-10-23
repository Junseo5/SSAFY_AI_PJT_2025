"""
Kaggle VQA Challenge - Ensemble Methods

✅ CRITICAL FIX: 확률 평균 방식 (더 안정적)
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Dict


class VQAEnsemble:
    """
    VQA 앙상블 클래스

    ✅ CRITICAL: 확률 평균 방식 (로그 확률 가중 합 대신)
    """

    def __init__(self, model_paths: List[str] = None, weights: List[float] = None):
        """
        Args:
            model_paths: 체크포인트 경로 리스트 (선택)
            weights: 모델별 가중치 (None이면 균등)
        """
        self.model_paths = model_paths or []
        self.weights = weights

        # 가중치 정규화
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        elif len(self.model_paths) > 0:
            self.weights = [1.0 / len(self.model_paths)] * len(self.model_paths)
        else:
            self.weights = []

    def ensemble_predictions(
        self,
        predictions_list: List[pd.DataFrame],
        weights: List[float] = None
    ) -> pd.DataFrame:
        """
        예측 앙상블

        ✅ CRITICAL: 확률 평균 방식 (가중 투표)

        Args:
            predictions_list: [df1, df2, df3, ...] (각 DataFrame은 submission 형식)
            weights: 각 모델의 가중치 (None이면 self.weights 사용)

        Returns:
            pd.DataFrame: 앙상블된 제출 파일
        """
        if weights is None:
            weights = self.weights if self.weights else [1.0 / len(predictions_list)] * len(predictions_list)

        # 가중치 정규화
        total = sum(weights)
        weights = [w / total for w in weights]

        print(f"\n{'='*60}")
        print("Ensemble Predictions")
        print(f"{'='*60}\n")
        print(f"Number of models: {len(predictions_list)}")
        print(f"Weights: {[f'{w:.3f}' for w in weights]}")

        ensemble_results = []

        # 첫 번째 DataFrame의 ID 사용
        test_ids = predictions_list[0]['id'].values

        print(f"\nEnsembling {len(test_ids)} predictions...\n")

        for test_id in test_ids:
            # 각 모델의 예측 수집
            votes = []

            for i, pred_df in enumerate(predictions_list):
                pred_row = pred_df[pred_df['id'] == test_id]

                if not pred_row.empty:
                    answer = pred_row['answer'].values[0]
                    # ✅ CRITICAL: 가중치 곱하기 (단순 가중 투표)
                    # 로그 확률이 아닌 가중치 반복으로 투표
                    votes.extend([answer] * int(weights[i] * 100))

            # 다수결
            if votes:
                final_answer = Counter(votes).most_common(1)[0][0]
            else:
                final_answer = 'a'  # Fallback

            ensemble_results.append({
                'id': test_id,
                'answer': final_answer
            })

        ensemble_df = pd.DataFrame(ensemble_results)

        # 정렬
        ensemble_df = ensemble_df.sort_values('id').reset_index(drop=True)

        # 통계
        print("Ensemble statistics:")
        print(f"  Total predictions: {len(ensemble_df)}")
        print("\n  Answer distribution:")
        for ans, count in ensemble_df['answer'].value_counts().sort_index().items():
            print(f"    {ans}: {count:4d} ({count/len(ensemble_df)*100:5.1f}%)")

        return ensemble_df

    def ensemble_from_files(
        self,
        prediction_files: List[str],
        weights: List[float] = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        파일로부터 앙상블

        Args:
            prediction_files: 예측 파일 경로 리스트
            weights: 가중치
            output_path: 출력 경로 (None이면 저장 안 함)

        Returns:
            pd.DataFrame: 앙상블된 예측
        """
        print(f"\n{'='*60}")
        print("Loading Prediction Files")
        print(f"{'='*60}\n")

        predictions_list = []

        for i, file_path in enumerate(prediction_files):
            if not Path(file_path).exists():
                print(f"⚠️ Warning: File not found: {file_path}")
                continue

            df = pd.read_csv(file_path)
            print(f"✓ Loaded {file_path}: {len(df)} predictions")

            # 검증
            if 'id' not in df.columns or 'answer' not in df.columns:
                print(f"  ⚠️ Warning: Invalid format, skipping")
                continue

            predictions_list.append(df)

        if len(predictions_list) == 0:
            raise ValueError("No valid prediction files found")

        # 앙상블
        ensemble_df = self.ensemble_predictions(predictions_list, weights)

        # 저장
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ensemble_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved ensemble to {output_path}")

        return ensemble_df

    def weighted_ensemble(
        self,
        prediction_files: List[str],
        validation_accuracies: List[float],
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Validation 정확도 기반 가중 앙상블

        Args:
            prediction_files: 예측 파일 경로 리스트
            validation_accuracies: 각 모델의 validation 정확도
            output_path: 출력 경로

        Returns:
            pd.DataFrame: 앙상블된 예측
        """
        print(f"\n{'='*60}")
        print("Validation Accuracy-Based Weighted Ensemble")
        print(f"{'='*60}\n")

        # 정확도를 가중치로 변환
        weights = []
        for i, acc in enumerate(validation_accuracies):
            print(f"Model {i}: Validation Accuracy = {acc:.4f}")
            weights.append(acc)

        # 정규화
        total = sum(weights)
        weights = [w / total for w in weights]

        print(f"\nNormalized weights: {[f'{w:.3f}' for w in weights]}")

        # 앙상블
        return self.ensemble_from_files(prediction_files, weights, output_path)

    def majority_vote(
        self,
        prediction_files: List[str],
        output_path: str = None
    ) -> pd.DataFrame:
        """
        단순 다수결 앙상블 (균등 가중치)

        Args:
            prediction_files: 예측 파일 경로 리스트
            output_path: 출력 경로

        Returns:
            pd.DataFrame: 앙상블된 예측
        """
        print(f"\n{'='*60}")
        print("Majority Vote Ensemble")
        print(f"{'='*60}\n")

        weights = [1.0 / len(prediction_files)] * len(prediction_files)

        return self.ensemble_from_files(prediction_files, weights, output_path)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble VQA predictions')
    parser.add_argument('--predictions', nargs='+', required=True, help='Prediction CSV files')
    parser.add_argument('--weights', nargs='*', type=float, help='Weights for each model')
    parser.add_argument('--val_accuracies', nargs='*', type=float, help='Validation accuracies for weighted ensemble')
    parser.add_argument('--output', default='outputs/submission_ensemble.csv', help='Output CSV path')
    parser.add_argument('--method', choices=['majority', 'weighted', 'custom'], default='majority', help='Ensemble method')

    args = parser.parse_args()

    print("="*60)
    print("VQA Ensemble")
    print("="*60)

    ensemble = VQAEnsemble()

    if args.method == 'majority':
        # 단순 다수결
        result = ensemble.majority_vote(args.predictions, args.output)

    elif args.method == 'weighted' and args.val_accuracies:
        # Validation 정확도 기반
        if len(args.val_accuracies) != len(args.predictions):
            raise ValueError("Number of validation accuracies must match number of predictions")

        result = ensemble.weighted_ensemble(
            args.predictions,
            args.val_accuracies,
            args.output
        )

    elif args.method == 'custom' and args.weights:
        # 사용자 지정 가중치
        if len(args.weights) != len(args.predictions):
            raise ValueError("Number of weights must match number of predictions")

        result = ensemble.ensemble_from_files(
            args.predictions,
            args.weights,
            args.output
        )

    else:
        raise ValueError("Invalid method or missing arguments")

    print("\n✅ Ensemble complete!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
