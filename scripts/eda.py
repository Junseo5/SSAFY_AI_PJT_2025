"""
Kaggle VQA Challenge - Exploratory Data Analysis (EDA)

This script performs comprehensive EDA on the VQA dataset including:
- Question type classification
- Answer format analysis
- Distribution visualization
- Data quality checks
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re  # ✅ CRITICAL FIX: Added missing import
from pathlib import Path
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class VQADataAnalyzer:
    """VQA 데이터 분석기"""

    def __init__(self, train_csv_path: str = 'data/train.csv'):
        """
        Args:
            train_csv_path: 학습 데이터 CSV 경로
        """
        self.train_csv_path = train_csv_path
        self.df = None
        self.type_patterns = {
            'counting': r'몇|개수|수|how many|count',
            'color': r'색|색깔|color|무슨색',
            'ocr': r'글자|문자|숫자|번호|읽|text|number|write|written',
            'yesno': r'인가|입니까|\?$|있는가|맞는가|yes|no',
            'location': r'어디|위치|where|장소|place',
            'attribute': r'무엇|what|어떤|kind|which'
        }

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        print(f"📁 Loading data from {self.train_csv_path}...")
        self.df = pd.read_csv(self.train_csv_path)
        print(f"✓ Loaded {len(self.df)} samples")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst 3 rows:")
        print(self.df.head(3))
        return self.df

    def analyze_question_types(self) -> Dict[str, int]:
        """
        질문 유형 분류 및 분포 분석

        Returns:
            dict: {question_type: count}
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*60)
        print("📊 Question Type Analysis")
        print("="*60)

        def classify_question(question: str) -> str:
            """질문을 유형별로 분류"""
            if pd.isna(question):
                return 'unknown'

            question_lower = question.lower()

            for qtype, pattern in self.type_patterns.items():
                if re.search(pattern, question_lower, re.I):
                    return qtype
            return 'general'

        self.df['question_type'] = self.df['question'].apply(classify_question)

        type_counts = self.df['question_type'].value_counts().to_dict()

        print("\nQuestion Type Distribution:")
        for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            percentage = count / len(self.df) * 100
            print(f"  {qtype:12s}: {count:4d} ({percentage:5.1f}%)")

        return type_counts

    def analyze_answer_format(self) -> Dict[str, int]:
        """
        보기 형식 분석

        Returns:
            dict: {format_type: count}
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*60)
        print("📝 Answer Format Analysis")
        print("="*60)

        def classify_format(row) -> str:
            """보기 형식 분류"""
            # 4개의 보기를 결합
            choices = f"{row['a']} {row['b']} {row['c']} {row['d']}"

            if pd.isna(choices):
                return 'unknown'

            # 한글만
            if re.match(r'^[가-힣\s]+$', choices):
                return 'pure_korean'
            # 영어만
            elif re.match(r'^[a-zA-Z\s]+$', choices):
                return 'pure_english'
            # 숫자 포함
            elif re.search(r'\d', choices):
                return 'numeric'
            else:
                return 'mixed'

        self.df['answer_format'] = self.df.apply(classify_format, axis=1)

        format_counts = self.df['answer_format'].value_counts().to_dict()

        print("\nAnswer Format Distribution:")
        for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
            percentage = count / len(self.df) * 100
            print(f"  {fmt:15s}: {count:4d} ({percentage:5.1f}%)")

        return format_counts

    def analyze_answer_distribution(self) -> Dict[str, int]:
        """
        정답 분포 분석 (a/b/c/d)

        Returns:
            dict: {answer: count}
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*60)
        print("🎯 Answer Label Distribution")
        print("="*60)

        answer_counts = self.df['answer'].value_counts().to_dict()

        print("\nAnswer Distribution:")
        for ans, count in sorted(answer_counts.items()):
            percentage = count / len(self.df) * 100
            print(f"  {ans}: {count:4d} ({percentage:5.1f}%)")

        # 균형 체크
        expected = len(self.df) / 4
        max_deviation = max(abs(count - expected) for count in answer_counts.values())
        if max_deviation / expected > 0.2:
            print(f"\n⚠️  Warning: Answer distribution is imbalanced (max deviation: {max_deviation/expected*100:.1f}%)")
        else:
            print(f"\n✓ Answer distribution is balanced")

        return answer_counts

    def analyze_text_lengths(self) -> Tuple[Dict, Dict]:
        """
        텍스트 길이 분석

        Returns:
            tuple: (question_lengths, choice_lengths)
        """
        if self.df is None:
            self.load_data()

        print("\n" + "="*60)
        print("📏 Text Length Analysis")
        print("="*60)

        # 질문 길이
        self.df['question_length'] = self.df['question'].str.len()

        q_stats = {
            'min': self.df['question_length'].min(),
            'max': self.df['question_length'].max(),
            'mean': self.df['question_length'].mean(),
            'median': self.df['question_length'].median()
        }

        print(f"\nQuestion Length Statistics:")
        print(f"  Min:    {q_stats['min']:.0f}")
        print(f"  Max:    {q_stats['max']:.0f}")
        print(f"  Mean:   {q_stats['mean']:.1f}")
        print(f"  Median: {q_stats['median']:.0f}")

        # 보기 길이
        self.df['choice_a_length'] = self.df['a'].str.len()
        self.df['choice_b_length'] = self.df['b'].str.len()
        self.df['choice_c_length'] = self.df['c'].str.len()
        self.df['choice_d_length'] = self.df['d'].str.len()

        avg_choice_length = (
            self.df['choice_a_length'] +
            self.df['choice_b_length'] +
            self.df['choice_c_length'] +
            self.df['choice_d_length']
        ) / 4

        c_stats = {
            'min': avg_choice_length.min(),
            'max': avg_choice_length.max(),
            'mean': avg_choice_length.mean(),
            'median': avg_choice_length.median()
        }

        print(f"\nAverage Choice Length Statistics:")
        print(f"  Min:    {c_stats['min']:.1f}")
        print(f"  Max:    {c_stats['max']:.1f}")
        print(f"  Mean:   {c_stats['mean']:.1f}")
        print(f"  Median: {c_stats['median']:.1f}")

        return q_stats, c_stats

    def visualize_distribution(self, output_dir: str = 'outputs'):
        """
        분포 시각화

        Args:
            output_dir: 출력 디렉토리
        """
        if self.df is None or 'question_type' not in self.df.columns:
            self.analyze_question_types()

        if 'answer_format' not in self.df.columns:
            self.analyze_answer_format()

        if 'question_length' not in self.df.columns:
            self.analyze_text_lengths()

        # 출력 디렉토리 생성
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        print(f"\n📊 Creating visualizations...")

        # 4개 서브플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VQA Dataset Analysis', fontsize=16, fontweight='bold')

        # 1. 질문 유형 분포
        type_counts = self.df['question_type'].value_counts()
        axes[0, 0].bar(range(len(type_counts)), type_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_xticks(range(len(type_counts)))
        axes[0, 0].set_xticklabels(type_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Question Type Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. 정답 분포
        answer_counts = self.df['answer'].value_counts().sort_index()
        axes[0, 1].bar(answer_counts.index, answer_counts.values, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Answer Distribution')
        axes[0, 1].set_xlabel('Answer')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. 질문 길이 분포
        axes[1, 0].hist(self.df['question_length'], bins=30, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Question Length Distribution')
        axes[1, 0].set_xlabel('Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(self.df['question_length'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {self.df['question_length'].mean():.1f}")
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. 보기 형식 분포
        format_counts = self.df['answer_format'].value_counts()
        axes[1, 1].bar(range(len(format_counts)), format_counts.values, color='plum', edgecolor='black')
        axes[1, 1].set_xticks(range(len(format_counts)))
        axes[1, 1].set_xticklabels(format_counts.index, rotation=45, ha='right')
        axes[1, 1].set_title('Answer Format Distribution')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = Path(output_dir) / 'eda_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        plt.close()

    def check_data_quality(self):
        """데이터 품질 체크"""
        if self.df is None:
            self.load_data()

        print("\n" + "="*60)
        print("🔍 Data Quality Check")
        print("="*60)

        # 결측치 확인
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values")
        else:
            for col, count in missing[missing > 0].items():
                print(f"  ⚠️  {col}: {count} ({count/len(self.df)*100:.1f}%)")

        # 중복 확인
        print("\nDuplicate Rows:")
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("  ✓ No duplicate rows")
        else:
            print(f"  ⚠️  {duplicates} duplicate rows found")

        # ID 중복 확인
        if 'id' in self.df.columns:
            print("\nID Duplicates:")
            id_duplicates = self.df['id'].duplicated().sum()
            if id_duplicates == 0:
                print("  ✓ No duplicate IDs")
            else:
                print(f"  ⚠️  {id_duplicates} duplicate IDs found")

        # 정답 유효성 확인
        print("\nAnswer Validity:")
        valid_answers = self.df['answer'].isin(['a', 'b', 'c', 'd']).all()
        if valid_answers:
            print("  ✓ All answers are valid (a/b/c/d)")
        else:
            invalid_count = (~self.df['answer'].isin(['a', 'b', 'c', 'd'])).sum()
            print(f"  ⚠️  {invalid_count} invalid answers found")
            print(f"  Invalid values: {self.df[~self.df['answer'].isin(['a', 'b', 'c', 'd'])]['answer'].unique()}")

    def generate_summary_report(self, output_dir: str = 'outputs'):
        """요약 리포트 생성"""
        if self.df is None:
            self.load_data()

        output_path = Path(output_dir) / 'eda_summary.txt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("VQA Dataset Summary Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total Samples: {len(self.df)}\n\n")

            # 질문 유형
            if 'question_type' in self.df.columns:
                f.write("Question Type Distribution:\n")
                for qtype, count in self.df['question_type'].value_counts().items():
                    f.write(f"  {qtype:12s}: {count:4d} ({count/len(self.df)*100:5.1f}%)\n")
                f.write("\n")

            # 정답 분포
            f.write("Answer Distribution:\n")
            for ans, count in sorted(self.df['answer'].value_counts().items()):
                f.write(f"  {ans}: {count:4d} ({count/len(self.df)*100:5.1f}%)\n")
            f.write("\n")

            # 텍스트 길이
            if 'question_length' in self.df.columns:
                f.write("Text Length Statistics:\n")
                f.write(f"  Question Length (mean): {self.df['question_length'].mean():.1f}\n")
                f.write(f"  Question Length (range): {self.df['question_length'].min():.0f} - {self.df['question_length'].max():.0f}\n")
                f.write("\n")

        print(f"\n✓ Saved summary report to {output_path}")

    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n" + "="*60)
        print("🚀 Running Full VQA Data Analysis")
        print("="*60 + "\n")

        # 1. 데이터 로드
        self.load_data()

        # 2. 질문 유형 분석
        self.analyze_question_types()

        # 3. 답변 형식 분석
        self.analyze_answer_format()

        # 4. 정답 분포 분석
        self.analyze_answer_distribution()

        # 5. 텍스트 길이 분석
        self.analyze_text_lengths()

        # 6. 데이터 품질 체크
        self.check_data_quality()

        # 7. 시각화
        self.visualize_distribution()

        # 8. 요약 리포트
        self.generate_summary_report()

        print("\n" + "="*60)
        print("✅ EDA Complete!")
        print("="*60)
        print("\nGenerated files:")
        print("  - outputs/eda_distribution.png")
        print("  - outputs/eda_summary.txt")
        print("\nEnhanced DataFrame saved with columns:")
        print("  - question_type")
        print("  - answer_format")
        print("  - question_length")

        return self.df


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='VQA Dataset EDA')
    parser.add_argument('--train_csv', default='data/train.csv', help='Path to train.csv')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')

    args = parser.parse_args()

    # 분석 실행
    analyzer = VQADataAnalyzer(train_csv_path=args.train_csv)
    df_enhanced = analyzer.run_full_analysis()

    # 강화된 DataFrame 저장 (선택사항)
    enhanced_path = 'data/train_with_types.csv'
    df_enhanced.to_csv(enhanced_path, index=False)
    print(f"\n✓ Enhanced DataFrame saved to {enhanced_path}")


if __name__ == "__main__":
    main()
