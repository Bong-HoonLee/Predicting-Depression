import torch
from torch import tensor


target = tensor([0, 1, 0, 1, 0, 1])
preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])

#1
from torchmetrics.classification import BinaryAccuracy

metric = BinaryAccuracy()
print(f"built-in binary_accuracy: {metric(preds, target):.4f}")


#2
def apply_threshold(y_pred, y_true, threshold=0.5):
    # 두 값의 절대 차이 계산
    difference = torch.abs(y_pred - y_true)
    
    # 차이가 threshold 이내인 경우를 찾음
    return (difference <= threshold).int()

def custom_accuracy(y_pred, y_true, threshold=0.5):
    # 예측값과 실제값의 차이 계산
    difference = torch.abs(y_pred - y_true)
    
    # 차이가 threshold 이하인 경우를 정확하게 예측한 것으로 간주
    correct_predictions = (difference <= threshold).sum().item()
    
    # 전체 예측의 수
    total_predictions = y_pred.numel()
    
    # 정확도 계산
    return correct_predictions / total_predictions

# 정확도 계산
acc = custom_accuracy(preds, target)
print(f'custom accuracy: {acc:.4f}')

