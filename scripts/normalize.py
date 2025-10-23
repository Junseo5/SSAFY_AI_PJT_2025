"""
Kaggle VQA Challenge - Answer Normalization

This script handles answer normalization including:
- Korean/English number normalization
- Special character handling
- Unit conversion
- Consistent formatting
"""

import re
import unicodedata
from typing import Dict
import yaml
from pathlib import Path


class AnswerNormalizer:
    """답변 정규화 클래스"""

    def __init__(self, config_path: str = 'config/normalize.yaml'):
        """
        Args:
            config_path: 정규화 규칙 설정 파일 경로
        """
        self.config_path = config_path
        self.rules = self._load_or_create_config()

    def _load_or_create_config(self) -> Dict:
        """설정 파일 로드 또는 생성"""
        config_file = Path(self.config_path)

        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 기본 규칙 생성
            default_rules = self._create_default_rules()

            # 디렉토리 생성
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # 저장
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_rules, f, allow_unicode=True, default_flow_style=False)

            print(f"✓ Created default normalization config at {config_file}")
            return default_rules

    def _create_default_rules(self) -> Dict:
        """기본 정규화 규칙 생성"""
        return {
            'korean_numbers': {
                '하나': '1', '둘': '2', '셋': '3', '넷': '4', '다섯': '5',
                '여섯': '6', '일곱': '7', '여덟': '8', '아홉': '9', '열': '10',
                '한': '1', '두': '2', '세': '3', '네': '4',
                '영': '0', '공': '0', '일': '1', '이': '2', '삼': '3',
                '사': '4', '오': '5', '육': '6', '칠': '7', '팔': '8', '구': '9', '십': '10'
            },
            'units': {
                '미터': 'm', '센티미터': 'cm', '킬로미터': 'km',
                '그램': 'g', '킬로그램': 'kg',
                '개': '', '마리': '', '명': '', '대': ''
            },
            'punctuation': {
                '·': '.', '˙': '.', ',': '', '、': '',
                ''': "'", ''': "'", '"': '"', '"': '"'
            },
            'whitespace': {
                'normalize': True,
                'strip': True,
                'collapse': True
            },
            'case': {
                'lowercase_answers': False,  # a/b/c/d는 소문자 유지
                'normalize_korean': True
            }
        }

    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화

        Args:
            text: 입력 텍스트

        Returns:
            str: 정규화된 텍스트
        """
        if not isinstance(text, str) or text == '':
            return text

        # 1. Unicode 정규화 (NFKC)
        text = unicodedata.normalize('NFKC', text)

        # 2. 한글 숫자 변환
        if self.rules.get('korean_numbers'):
            for kr_num, arab_num in self.rules['korean_numbers'].items():
                # 단어 경계에서만 변환
                text = re.sub(rf'\b{kr_num}\b', arab_num, text)

        # 3. 단위 정규화
        if self.rules.get('units'):
            for unit, normalized in self.rules['units'].items():
                text = text.replace(unit, normalized)

        # 4. 구두점 정규화
        if self.rules.get('punctuation'):
            for punct, normalized in self.rules['punctuation'].items():
                text = text.replace(punct, normalized)

        # 5. 공백 정규화
        if self.rules.get('whitespace', {}).get('collapse', True):
            text = re.sub(r'\s+', ' ', text)

        if self.rules.get('whitespace', {}).get('strip', True):
            text = text.strip()

        return text

    def normalize_answer(self, answer: str) -> str:
        """
        정답 레이블 정규화 (a/b/c/d)

        Args:
            answer: 정답 레이블

        Returns:
            str: 정규화된 정답 ('a', 'b', 'c', 'd')
        """
        if not isinstance(answer, str):
            return str(answer).lower().strip()

        # 소문자 변환 및 공백 제거
        answer = answer.lower().strip()

        # 유효성 검증
        if answer in ['a', 'b', 'c', 'd']:
            return answer
        else:
            print(f"⚠️  Warning: Invalid answer '{answer}', defaulting to 'a'")
            return 'a'

    def normalize_question(self, question: str) -> str:
        """
        질문 정규화

        Args:
            question: 질문 텍스트

        Returns:
            str: 정규화된 질문
        """
        return self.normalize_text(question)

    def normalize_choice(self, choice: str) -> str:
        """
        보기 정규화

        Args:
            choice: 보기 텍스트

        Returns:
            str: 정규화된 보기
        """
        return self.normalize_text(choice)

    def normalize_row(self, row: Dict) -> Dict:
        """
        전체 행 정규화

        Args:
            row: {'question': ..., 'a': ..., 'b': ..., 'c': ..., 'd': ..., 'answer': ...}

        Returns:
            dict: 정규화된 행
        """
        normalized = row.copy()

        # 질문 정규화
        if 'question' in normalized:
            normalized['question'] = self.normalize_question(normalized['question'])

        # 보기 정규화
        for choice in ['a', 'b', 'c', 'd']:
            if choice in normalized:
                normalized[choice] = self.normalize_choice(normalized[choice])

        # 정답 정규화
        if 'answer' in normalized:
            normalized['answer'] = self.normalize_answer(normalized['answer'])

        return normalized

    def batch_normalize(self, df):
        """
        DataFrame 일괄 정규화

        Args:
            df: pandas DataFrame

        Returns:
            DataFrame: 정규화된 DataFrame
        """
        import pandas as pd

        print(f"🔄 Normalizing {len(df)} rows...")

        # 질문 정규화
        if 'question' in df.columns:
            df['question'] = df['question'].apply(self.normalize_question)

        # 보기 정규화
        for choice in ['a', 'b', 'c', 'd']:
            if choice in df.columns:
                df[choice] = df[choice].apply(self.normalize_choice)

        # 정답 정규화
        if 'answer' in df.columns:
            df['answer'] = df['answer'].apply(self.normalize_answer)

        print("✓ Normalization complete")
        return df


def create_normalization_config():
    """정규화 설정 파일 생성 (독립 실행용)"""
    normalizer = AnswerNormalizer()
    print(f"✓ Normalization config created at {normalizer.config_path}")
    print("\nDefault rules:")
    print(yaml.dump(normalizer.rules, allow_unicode=True, default_flow_style=False))


def main():
    """메인 실행 함수"""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Normalize VQA dataset')
    parser.add_argument('--input_csv', default='data/train.csv', help='Input CSV file')
    parser.add_argument('--output_csv', default='data/train_normalized.csv', help='Output CSV file')
    parser.add_argument('--config', default='config/normalize.yaml', help='Normalization config file')
    parser.add_argument('--create_config', action='store_true', help='Create default config and exit')

    args = parser.parse_args()

    if args.create_config:
        create_normalization_config()
        return

    # 정규화 실행
    print("="*60)
    print("VQA Answer Normalization")
    print("="*60 + "\n")

    normalizer = AnswerNormalizer(config_path=args.config)

    print(f"📁 Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"✓ Loaded {len(df)} samples\n")

    # 정규화 전 샘플
    print("Before normalization (first 3 rows):")
    print(df[['question', 'a', 'b', 'c', 'd', 'answer']].head(3))
    print()

    # 정규화
    df_normalized = normalizer.batch_normalize(df)

    # 정규화 후 샘플
    print("\nAfter normalization (first 3 rows):")
    print(df_normalized[['question', 'a', 'b', 'c', 'd', 'answer']].head(3))
    print()

    # 저장
    df_normalized.to_csv(args.output_csv, index=False)
    print(f"✓ Saved normalized data to {args.output_csv}")


if __name__ == "__main__":
    main()
