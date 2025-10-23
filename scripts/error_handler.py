"""
Kaggle VQA Challenge - Error Handler

This script provides robust error handling for:
- GPU Out of Memory (OOM)
- Tokenization errors (Korean text)
- Model loading failures
- Safe inference with retries
"""

import logging
import torch
import time
import unicodedata
from typing import Callable, Any
from functools import wraps

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VQAErrorHandler:
    """VQA 프로젝트 전용 에러 핸들러"""

    @staticmethod
    def handle_gpu_oom(func: Callable) -> Callable:
        """
        GPU OOM 자동 복구 데코레이터

        Args:
            func: 래핑할 함수

        Returns:
            Callable: 래핑된 함수

        Usage:
            @VQAErrorHandler.handle_gpu_oom
            def train_epoch(model, dataloader, batch_size=4):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e).lower()

                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.warning("⚠️  GPU OOM detected. Clearing cache...")

                    # GPU 캐시 정리
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Batch size 줄이기 시도
                    if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                        new_batch_size = max(1, kwargs['batch_size'] // 2)
                        logger.info(
                            f"🔄 Reducing batch_size: {kwargs['batch_size']} → {new_batch_size}"
                        )
                        kwargs['batch_size'] = new_batch_size

                        # 재시도
                        try:
                            return func(*args, **kwargs)
                        except RuntimeError as e2:
                            logger.error(f"❌ Retry failed even with reduced batch size: {e2}")
                            raise
                    else:
                        logger.error("❌ Cannot reduce batch_size further")
                        raise
                else:
                    # 다른 RuntimeError는 그대로 전파
                    raise

        return wrapper

    @staticmethod
    def handle_tokenization_error(func: Callable) -> Callable:
        """
        한글 토큰화 오류 방지 데코레이터

        Args:
            func: 래핑할 함수

        Returns:
            Callable: 래핑된 함수

        Usage:
            @VQAErrorHandler.handle_tokenization_error
            def tokenize_text(text):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()

                if "encode" in error_msg or "token" in error_msg or "unicode" in error_msg:
                    logger.error(f"❌ Tokenization failed: {e}")

                    # 텍스트 정규화 재시도
                    if 'text' in kwargs:
                        logger.info("🔄 Retrying with normalized text...")

                        # NFKC 정규화
                        kwargs['text'] = unicodedata.normalize('NFKC', kwargs['text'])

                        # 제어 문자 제거
                        import re
                        kwargs['text'] = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', kwargs['text'])

                        try:
                            return func(*args, **kwargs)
                        except Exception as e2:
                            logger.error(f"❌ Retry failed with normalized text: {e2}")
                            raise
                    else:
                        raise
                else:
                    raise

        return wrapper

    @staticmethod
    def handle_model_load_error(func: Callable) -> Callable:
        """
        모델 로딩 실패 처리 데코레이터

        Args:
            func: 래핑할 함수

        Returns:
            Callable: 래핑된 함수

        Usage:
            @VQAErrorHandler.handle_model_load_error
            def load_model(model_path):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()

                if "load" in error_msg or "checkpoint" in error_msg or "state_dict" in error_msg:
                    logger.error(f"❌ Model loading failed: {e}")

                    # 백업 체크포인트 시도
                    if 'model_path' in kwargs:
                        original_path = kwargs['model_path']

                        # fold 번호를 바꿔서 시도
                        backup_paths = []
                        for fold in range(3):
                            backup_path = original_path.replace('fold0', f'fold{fold}')
                            if backup_path != original_path:
                                backup_paths.append(backup_path)

                        for backup_path in backup_paths:
                            logger.info(f"🔄 Trying backup checkpoint: {backup_path}")
                            kwargs['model_path'] = backup_path

                            try:
                                return func(*args, **kwargs)
                            except Exception:
                                continue

                        logger.error("❌ All backup checkpoints failed")

                    raise
                else:
                    raise

        return wrapper

    @staticmethod
    def safe_inference(
        predictor,
        image_path: str,
        question: str,
        choices: dict,
        max_retries: int = 3
    ) -> dict:
        """
        안전한 추론 (재시도 포함)

        Args:
            predictor: VQAPredictor 인스턴스
            image_path: 이미지 경로
            question: 질문
            choices: 선택지
            max_retries: 최대 재시도 횟수

        Returns:
            dict: {
                'prediction': str,
                'confidence': float
            }
        """
        for attempt in range(max_retries):
            try:
                result = predictor.predict(image_path, question, choices)

                # 결과 검증
                if 'prediction' not in result:
                    raise ValueError("No prediction in result")

                if result['prediction'] not in ['a', 'b', 'c', 'd']:
                    raise ValueError(f"Invalid prediction: {result['prediction']}")

                return result

            except Exception as e:
                logger.warning(
                    f"⚠️  Inference attempt {attempt+1}/{max_retries} failed: {e}"
                )

                if attempt == max_retries - 1:
                    logger.error(f"❌ All retries failed for image: {image_path}")

                    # 폴백: 무작위 답 반환
                    import random
                    fallback = random.choice(['a', 'b', 'c', 'd'])
                    logger.warning(f"🎲 Using fallback prediction: {fallback}")

                    return {
                        'prediction': fallback,
                        'confidence': 0.0,
                        'fallback': True
                    }

                # 재시도 전 대기
                time.sleep(1 * (attempt + 1))  # 지수 백오프

        # 도달하지 않음
        raise RuntimeError("Unexpected: safe_inference reached end without return")

    @staticmethod
    def clear_gpu_cache():
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            logger.info("✓ GPU cache cleared")

    @staticmethod
    def get_gpu_memory_stats():
        """GPU 메모리 통계"""
        if not torch.cuda.is_available():
            return {}

        stats = {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
        }

        return stats

    @staticmethod
    def print_gpu_memory_stats():
        """GPU 메모리 통계 출력"""
        stats = VQAErrorHandler.get_gpu_memory_stats()

        if stats:
            print(f"\n{'─'*60}")
            print("GPU Memory Stats:")
            print(f"{'─'*60}")
            print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
            print(f"  Max Used:  {stats['max_allocated_gb']:.2f} GB")
            print(f"{'─'*60}\n")


# 사용 예시
if __name__ == "__main__":
    # 데코레이터 사용 예시

    @VQAErrorHandler.handle_gpu_oom
    @VQAErrorHandler.handle_tokenization_error
    def example_training_step(model, batch, batch_size=4):
        """예시 학습 스텝"""
        # 학습 코드
        print(f"Training with batch_size={batch_size}")
        return "success"

    # 테스트
    print("="*60)
    print("VQA Error Handler - Examples")
    print("="*60)

    # GPU 메모리 통계
    VQAErrorHandler.print_gpu_memory_stats()

    # GPU 캐시 정리
    VQAErrorHandler.clear_gpu_cache()

    print("\n✓ Error handler loaded successfully")
    print("Usage: Decorate functions with @VQAErrorHandler.handle_gpu_oom, etc.")
