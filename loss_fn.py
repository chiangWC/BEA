import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


def task_sim_loss(y_true, y_pred):
    return 

def total_loss(y_true, y_pred):
    """
    y_true: Tensor, shape [batch_size, 4], int labels (0,1,2)
    y_pred: Tensor, shape [batch_size, 4, 3], raw logits
    """
    y_true_np = y_true.cpu().numpy()  # 转为 numpy，shape [B, 4]

    total_loss = 0.0
    for task_id in range(4):
        y_task = y_true_np[:, task_id]
        classes = np.unique(y_task)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_task)

        full_weights = np.ones(3, dtype=np.float32)
        for cls_id, w in zip(classes, weights):
            full_weights[int(cls_id)] = w

        weight_task = torch.tensor(full_weights, dtype=torch.float32).to(y_pred.device)
        weight_task = weight_task.clamp(min=0.55, max=5.0)

        pred_task = y_pred[:, task_id, :]   # [B, 3]
        label_task = y_true[:, task_id]     # [B]

        loss_fn = nn.CrossEntropyLoss(weight=weight_task)
        task_loss = loss_fn(pred_task, label_task)

        total_loss += task_loss

    return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # shape: [num_classes], optional
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def per_class_accuracy(y_true, y_pred, num_classes=3):
    acc_dict = {}

    for cls in range(num_classes):
        # 找到属于该类的索引
        cls_indices = np.where(y_true == cls)[0]
        if len(cls_indices) == 0:
            acc_dict[cls] = None  # 没有该类样本，无法计算准确率
            continue
        # 计算该类的准确率
        cls_acc = accuracy_score(y_true[cls_indices], y_pred[cls_indices])
        acc_dict[cls] = round(float(cls_acc), 4)

    return acc_dict


def evaluate(references, predictions):
    predictions = np.argmax(predictions, axis=-1) 

    eval_rst = []
    for ii in range(predictions.shape[1]):
        acc_dict = per_class_accuracy(references[:, ii], predictions[:, ii])
        print(f'task_{ii+1}: {acc_dict}')

        ex_acc = accuracy_score(references[:, ii], predictions[:, ii])
        ex_f1 = f1_score(references[:, ii], predictions[:, ii], average='macro')

        y_true_len = np.where(references[:, ii] == 0, 0, 1)
        y_pred_len = np.where(predictions[:, ii] == 0, 0, 1)
        len_acc = accuracy_score(y_true_len, y_pred_len)
        len_f1 = f1_score(y_true_len, y_pred_len, average='macro')

        eval_rst.append({
            'ex_macro_f1': round(float(ex_f1), 4),
            'ex_acc': round(float(ex_acc), 4),
            'len_macro_f1': round(float(len_f1), 4),
            'len_acc': round(float(len_acc), 4)
        })

    return eval_rst