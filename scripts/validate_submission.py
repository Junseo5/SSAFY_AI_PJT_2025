"""
Kaggle VQA Challenge - Submission Validation

제출 파일 형식 검증
"""

import pandas as pd
import sys
from pathlib import Path


def validate_submission(file_path: str, test_csv_path: str = 'data/test.csv') -> bool:
    """
    제출 파일 검증

    ✅ 엄격한 검증 규칙 적용

    Args:
        file_path: 제출 파일 경로
        test_csv_path: 테스트 데이터 경로

    Returns:
        bool: 검증 통과 여부
    """
    print(f"\n{'='*60}")
    print("Submission File Validation")
    print(f"{'='*60}\n")
    print(f"File: {file_path}")

    try:
        # 파일 존재 확인
        if not Path(file_path).exists():
            print(f"❌ FAILED: File does not exist")
            return False

        # CSV 로드
        df = pd.read_csv(file_path)
        print(f"✓ File loaded successfully ({len(df)} rows)")

        # 1. 컬럼 확인
        print(f"\n[1/8] Checking columns...")
        expected_columns = ['id', 'answer']
        actual_columns = list(df.columns)

        if actual_columns != expected_columns:
            print(f"❌ FAILED: Invalid columns")
            print(f"  Expected: {expected_columns}")
            print(f"  Actual:   {actual_columns}")
            return False

        print(f"✓ Columns are correct: {expected_columns}")

        # 2. 답 형식 확인 (소문자만)
        print(f"\n[2/8] Checking answer format...")
        valid_answers = ['a', 'b', 'c', 'd']

        invalid_answers = df[~df['answer'].isin(valid_answers)]
        if len(invalid_answers) > 0:
            print(f"❌ FAILED: Invalid answers found ({len(invalid_answers)} rows)")
            print(f"  Invalid values: {invalid_answers['answer'].unique()}")
            print(f"\n  First 5 invalid rows:")
            print(invalid_answers.head())
            return False

        print(f"✓ All answers are valid (a/b/c/d)")

        # 3. 공백 확인
        print(f"\n[3/8] Checking for whitespace...")
        has_whitespace = df['answer'].str.contains(' ', na=False).any()

        if has_whitespace:
            print(f"❌ FAILED: Whitespace found in answers")
            rows_with_ws = df[df['answer'].str.contains(' ', na=False)]
            print(f"  Affected rows: {len(rows_with_ws)}")
            print(rows_with_ws.head())
            return False

        print(f"✓ No whitespace in answers")

        # 4. 대문자 확인
        print(f"\n[4/8] Checking for uppercase...")
        has_uppercase = df['answer'].str.contains('[A-D]', na=False, regex=True).any()

        if has_uppercase:
            print(f"❌ FAILED: Uppercase letters found")
            rows_with_upper = df[df['answer'].str.contains('[A-D]', na=False, regex=True)]
            print(f"  Affected rows: {len(rows_with_upper)}")
            print(rows_with_upper.head())
            return False

        print(f"✓ No uppercase letters")

        # 5. ID 중복 확인
        print(f"\n[5/8] Checking for duplicate IDs...")
        duplicates = df['id'].duplicated()

        if duplicates.any():
            print(f"❌ FAILED: Duplicate IDs found")
            dup_ids = df[duplicates]['id'].values
            print(f"  Duplicate IDs: {dup_ids}")
            return False

        print(f"✓ No duplicate IDs")

        # 6. 결측치 확인
        print(f"\n[6/8] Checking for missing values...")
        missing = df.isnull().sum()

        if missing.sum() > 0:
            print(f"❌ FAILED: Missing values found")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} missing")
            return False

        print(f"✓ No missing values")

        # 7. 모든 test ID 포함 확인
        print(f"\n[7/8] Checking test IDs...")
        if Path(test_csv_path).exists():
            test_df = pd.read_csv(test_csv_path)
            expected_ids = set(test_df['id'].values)
            actual_ids = set(df['id'].values)

            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids

            if missing_ids:
                print(f"❌ FAILED: Missing IDs ({len(missing_ids)})")
                print(f"  First 10 missing: {list(missing_ids)[:10]}")
                return False

            if extra_ids:
                print(f"❌ FAILED: Extra IDs ({len(extra_ids)})")
                print(f"  First 10 extra: {list(extra_ids)[:10]}")
                return False

            print(f"✓ All test IDs present ({len(expected_ids)} IDs)")
        else:
            print(f"⚠️  Warning: Test CSV not found, skipping ID check")

        # 8. 데이터 타입 확인
        print(f"\n[8/8] Checking data types...")

        # ID는 정수여야 함
        if not pd.api.types.is_integer_dtype(df['id']):
            try:
                df['id'] = df['id'].astype(int)
            except:
                print(f"❌ FAILED: ID column is not integer-compatible")
                return False

        # Answer는 문자열이어야 함
        if df['answer'].dtype != 'object':
            print(f"⚠️  Warning: Answer dtype is {df['answer'].dtype}, converting to object")
            df['answer'] = df['answer'].astype(str)

        print(f"✓ Data types are correct")

        # 최종 통계
        print(f"\n{'='*60}")
        print("Validation Summary")
        print(f"{'='*60}\n")
        print(f"Total predictions: {len(df)}")
        print(f"\nAnswer distribution:")
        for ans, count in df['answer'].value_counts().sort_index().items():
            percentage = count / len(df) * 100
            print(f"  {ans}: {count:5d} ({percentage:5.1f}%)")

        # 균형 체크
        expected_per_answer = len(df) / 4
        max_deviation = max(abs(count - expected_per_answer) for count in df['answer'].value_counts().values)

        if max_deviation / expected_per_answer > 0.3:
            print(f"\n⚠️  Warning: Answer distribution is imbalanced")
            print(f"  Max deviation from expected: {max_deviation/expected_per_answer*100:.1f}%")
        else:
            print(f"\n✓ Answer distribution is reasonable")

        print(f"\n{'='*60}")
        print("✅ VALIDATION PASSED!")
        print(f"{'='*60}\n")
        print(f"Your submission file is valid and ready for upload.")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ VALIDATION FAILED")
        print(f"{'='*60}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_common_issues(file_path: str, output_path: str = None):
    """
    일반적인 문제 자동 수정

    Args:
        file_path: 원본 파일 경로
        output_path: 수정된 파일 저장 경로 (None이면 원본 덮어쓰기)
    """
    print(f"\n{'='*60}")
    print("Attempting to Fix Common Issues")
    print(f"{'='*60}\n")

    df = pd.read_csv(file_path)
    fixed = False

    # 1. 대문자 → 소문자
    if df['answer'].str.contains('[A-D]', na=False, regex=True).any():
        print("🔧 Fixing: Converting uppercase to lowercase")
        df['answer'] = df['answer'].str.lower()
        fixed = True

    # 2. 공백 제거
    if df['answer'].str.contains(' ', na=False).any():
        print("🔧 Fixing: Removing whitespace")
        df['answer'] = df['answer'].str.strip()
        fixed = True

    # 3. ID 정렬
    if not df['id'].is_monotonic_increasing:
        print("🔧 Fixing: Sorting by ID")
        df = df.sort_values('id').reset_index(drop=True)
        fixed = True

    # 4. 중복 제거 (첫 번째 유지)
    if df['id'].duplicated().any():
        print("🔧 Fixing: Removing duplicate IDs (keeping first)")
        df = df.drop_duplicates(subset='id', keep='first')
        fixed = True

    if fixed:
        output_path = output_path or file_path
        df.to_csv(output_path, index=False)
        print(f"\n✓ Fixed submission saved to: {output_path}")
        return True
    else:
        print("\n✓ No issues found to fix")
        return False


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate submission file')
    parser.add_argument('--file', required=True, help='Submission CSV file')
    parser.add_argument('--test_csv', default='data/test.csv', help='Test CSV for ID validation')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix common issues')
    parser.add_argument('--output', help='Output path for fixed file')

    args = parser.parse_args()

    # 수정 모드
    if args.fix:
        fixed = fix_common_issues(args.file, args.output)
        if fixed:
            # 수정 후 재검증
            file_to_validate = args.output or args.file
            validate_submission(file_to_validate, args.test_csv)
    else:
        # 검증만
        is_valid = validate_submission(args.file, args.test_csv)

        if not is_valid:
            print("\n💡 Tip: Use --fix flag to attempt automatic fixes")
            sys.exit(1)


if __name__ == "__main__":
    main()
