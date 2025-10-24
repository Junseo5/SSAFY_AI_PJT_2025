"""
Kaggle_AllInOne_Pro2 개선 코드 모음
이 파일의 함수들을 노트북에 복사-붙여넣기하여 사용하세요.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
from pathlib import Path

# ============================================================================
# 1. 로깅 시스템
# ============================================================================

def setup_logging(log_dir, level=logging.INFO):
    """
    강화된 로깅 시스템 설정

    사용법:
        logger = setup_logging("/content/logs")
        logger.info("Training started")
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('VQA')
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f"{log_dir}/training_{timestamp}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# 2. 안전한 이미지 로드
# ============================================================================

def load_image_safe(img_path, fallback_size=512, logger=None):
    """
    안전한 이미지 로드 with fallback

    Args:
        img_path: 이미지 경로
        fallback_size: fallback 이미지 크기
        logger: 로거 (선택)

    Returns:
        (Image, success: bool)
    """
    try:
        img = Image.open(img_path).convert("RGB")
        # 이미지 검증
        if img.size[0] < 10 or img.size[1] < 10:
            raise ValueError(f"Image too small: {img.size}")
        return img, True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Image load failed ({img_path}): {e}")
        # Fallback: 흰색 이미지
        return Image.new('RGB', (fallback_size, fallback_size), color='white'), False


# ============================================================================
# 3. Direct Logits - 개선된 토큰 확률 계산
# ============================================================================

def get_choice_token_ids_robust(processor):
    """
    강화된 토큰 ID 추출 - 여러 변형 고려

    Args:
        processor: Huggingface processor

    Returns:
        dict: {choice: [token_ids]}
    """
    choice_tokens = {}
    for choice in ['a', 'b', 'c', 'd']:
        # 여러 변형 고려
        variants = [
            choice,           # "a"
            f" {choice}",     # " a"
            f"{choice} ",     # "a "
            choice.upper(),   # "A"
            f" {choice.upper()}",  # " A"
        ]
        all_token_ids = set()
        for variant in variants:
            try:
                token_ids = processor.tokenizer.encode(variant, add_special_tokens=False)
                all_token_ids.update(token_ids)
            except:
                pass
        choice_tokens[choice] = list(all_token_ids)
    return choice_tokens


def extract_choice_probs_enhanced(logits, choice_tokens):
    """
    개선된 choice 확률 추출

    Args:
        logits: [vocab_size] tensor
        choice_tokens: get_choice_token_ids_robust() 결과

    Returns:
        dict: {choice: probability}
    """
    choice_logits = {}
    for choice, token_ids in choice_tokens.items():
        if len(token_ids) > 0:
            # 여러 토큰의 logit 중 최대값 사용 (or 평균)
            max_logit = max([logits[tid].item() for tid in token_ids])
            choice_logits[choice] = max_logit
        else:
            choice_logits[choice] = -float('inf')

    # Softmax로 확률 변환
    logit_values = torch.tensor(list(choice_logits.values()))
    probs = F.softmax(logit_values, dim=0).numpy()

    return {choice: probs[i] for i, choice in enumerate(['a', 'b', 'c', 'd'])}


# ============================================================================
# 4. Temperature Scaling
# ============================================================================

class TemperatureScaler:
    """
    Temperature Scaling for Probability Calibration

    사용법:
        # 학습
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)  # val_logits: [N, 4], val_labels: [N] (0/1/2/3)

        # 적용
        calibrated_probs = scaler.transform(test_logits)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels, logger=None):
        """
        검증 세트로 최적 temperature 학습

        Args:
            logits: numpy array [N, 4] - raw logits for [a, b, c, d]
            labels: numpy array [N] - true labels (0=a, 1=b, 2=c, 3=d)
            logger: 로거 (선택)

        Returns:
            self
        """
        logits = np.array(logits)
        labels = np.array(labels)

        def nll_loss(temp):
            temp = max(temp[0], 0.01)  # 최소값 방지
            scaled_probs = F.softmax(torch.tensor(logits) / temp, dim=1).numpy()
            # Negative log-likelihood
            nll = -np.log(scaled_probs[np.arange(len(labels)), labels] + 1e-10).mean()
            return nll

        result = minimize(nll_loss, x0=[1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]

        if logger:
            logger.info(f"✅ Optimal Temperature: {self.temperature:.4f}")
        else:
            print(f"✅ Optimal Temperature: {self.temperature:.4f}")

        return self

    def transform(self, logits):
        """
        학습된 temperature로 확률 스케일

        Args:
            logits: numpy array [N, 4]

        Returns:
            numpy array [N, 4] - calibrated probabilities
        """
        logits = np.array(logits)
        return F.softmax(torch.tensor(logits) / self.temperature, dim=1).numpy()


# ============================================================================
# 5. 메모리 최적화
# ============================================================================

def clear_memory(logger=None):
    """
    메모리 정리

    사용법:
        for fold in range(3):
            train_fold(...)
            clear_memory(logger)
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9

        if logger:
            logger.info(f"💾 Memory cleared - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            print(f"💾 Memory cleared - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


# ============================================================================
# 6. 체크포인트 관리
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, fold, metrics, save_path, logger=None):
    """
    체크포인트 저장

    Args:
        model: 모델
        optimizer: 옵티마이저
        scheduler: 스케줄러
        epoch: 현재 에폭
        fold: 현재 fold
        metrics: dict - {'val_loss': x, 'val_acc': y}
        save_path: 저장 경로
        logger: 로거 (선택)
    """
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    torch.save(checkpoint, save_path)

    if logger:
        logger.info(f"💾 Checkpoint saved: {save_path}")
    else:
        print(f"💾 Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger=None):
    """
    체크포인트 로드

    Args:
        model: 모델
        optimizer: 옵티마이저
        scheduler: 스케줄러
        checkpoint_path: 체크포인트 경로
        logger: 로거 (선택)

    Returns:
        (epoch, fold, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if logger:
        logger.info(f"✅ Checkpoint loaded: epoch {checkpoint['epoch']}, fold {checkpoint['fold']}")
    else:
        print(f"✅ Checkpoint loaded: epoch {checkpoint['epoch']}, fold {checkpoint['fold']}")

    return checkpoint['epoch'], checkpoint['fold'], checkpoint['metrics']


# ============================================================================
# 7. Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Early Stopping 구현

    사용법:
        early_stopping = EarlyStopping(patience=2, mode='max')  # mode='max' for accuracy

        for epoch in range(epochs):
            val_acc = validate(...)
            if early_stopping(val_acc, model):
                print("Early stopping triggered")
                break
    """

    def __init__(self, patience=3, mode='min', delta=0.0, logger=None):
        """
        Args:
            patience: 개선이 없어도 기다릴 에폭 수
            mode: 'min' (loss) or 'max' (accuracy)
            delta: 개선으로 간주할 최소 변화량
            logger: 로거 (선택)
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.logger = logger

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda new, best: new < best - delta
        else:
            self.is_better = lambda new, best: new > best + delta

    def __call__(self, score, model=None):
        """
        Args:
            score: 현재 스코어 (loss or accuracy)
            model: 모델 (best 저장용, 선택)

        Returns:
            bool: True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.logger:
                self.logger.info(f"✅ New best score: {score:.4f}")
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(f"⚠️ No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.logger:
                    self.logger.info(f"🛑 Early stopping triggered!")
                return True

        return False


# ============================================================================
# 8. 앙상블 가중치 최적화
# ============================================================================

def optimize_ensemble_weights(predictions_list, val_labels, logger=None):
    """
    검증 세트로 최적 앙상블 가중치 탐색

    Args:
        predictions_list: List[np.array] - 각 fold의 확률 [N, 4]
        val_labels: np.array [N] - 정답 레이블 (0/1/2/3)
        logger: 로거 (선택)

    Returns:
        np.array - 최적 가중치

    사용법:
        weights = optimize_ensemble_weights([fold0_probs, fold1_probs, fold2_probs], val_labels)
        # 이후 test 시
        ensemble_probs = sum(w * pred for w, pred in zip(weights, test_predictions))
    """
    n_models = len(predictions_list)

    def negative_accuracy(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()  # Normalize

        # 가중 평균
        ensemble_preds = sum(w * pred for w, pred in zip(weights, predictions_list))
        ensemble_labels = ensemble_preds.argmax(axis=1)

        acc = (ensemble_labels == val_labels).mean()
        return -acc  # Minimize negative accuracy

    # 초기 가중치: 동일
    init_weights = np.ones(n_models) / n_models

    # 최적화
    result = minimize(
        negative_accuracy,
        init_weights,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n_models
    )

    # Normalize
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    if logger:
        logger.info(f"✅ Optimal weights: {optimal_weights}")
        logger.info(f"   Expected accuracy: {-result.fun:.4f}")
    else:
        print(f"✅ Optimal weights: {optimal_weights}")
        print(f"   Expected accuracy: {-result.fun:.4f}")

    return optimal_weights


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Pro2 Enhancement Functions")
    print("=" * 60)
    print()
    print("이 파일의 함수들을 노트북에 복사-붙여넣기하여 사용하세요:")
    print()
    print("1. setup_logging() - 로깅 시스템")
    print("2. load_image_safe() - 안전한 이미지 로드")
    print("3. get_choice_token_ids_robust() - 개선된 토큰 ID 추출")
    print("4. extract_choice_probs_enhanced() - 개선된 확률 계산")
    print("5. TemperatureScaler - Temperature scaling")
    print("6. clear_memory() - 메모리 정리")
    print("7. save_checkpoint(), load_checkpoint() - 체크포인트 관리")
    print("8. EarlyStopping - Early stopping")
    print("9. optimize_ensemble_weights() - 앙상블 가중치 최적화")
    print()
    print("=" * 60)
