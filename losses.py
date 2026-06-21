import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(name, **kwargs):
    """
       根据名称获取损失函数。

       Args:
           name (str): 损失函数名称，支持 "cross_entropy", "weighted_cross_entropy", "focal_loss"。
           kwargs: 其他参数，用于配置损失函数。

       Returns:
           torch.nn.Module: 损失函数实例。
    """
    if name == "cross_entropy":
        # 普通交叉熵损失
        return nn.CrossEntropyLoss()
    elif name == "weighted_cross_entropy":
        # return nn.BCEWithLogitsLoss(pos_weight=kwargs["pos_weight"])
        return nn.CrossEntropyLoss(weight=kwargs["weight"])
    elif name == "weighted_cross_entropy_smoothing":
        return WeightedCrossEntropyWithSmoothing(pos_weight=kwargs["pos_weight"],
                                                 smoothing=kwargs.get("smoothing", 0.1))
    elif name == "focal_loss":
        return FocalLoss(alpha=kwargs["alpha"], gamma=kwargs["gamma"])
    elif name == "supcon_loss":
        return SupConLoss(temperature=kwargs.get("temperature", 0.07))
    elif name == "SupCon_WCE":
        return SupCon_WCE_Loss(weight=kwargs["weight"], temperature=kwargs.get("temperature", 0.07), lambda1=kwargs.get("lambda1", 1), lambda2=kwargs.get("lambda2", 0.02))
        # return SupCon_WCE_Loss(weight=kwargs["weight"], temperature=kwargs.get("temperature", 0.07), lambda1=kwargs.get("lambda_wce", 1), lambda2=kwargs.get("lambda_supcon", 0.02)) EXPERIMENT 15 及之前 都是用的这行代码 lambda_supcon = 0.02
    else:
        raise ValueError(f"Unsupported loss: {name}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class WeightedCrossEntropyWithSmoothing(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        """
        带有 Label Smoothing 的 Weighted BCEWithLogitsLoss。
        Args:
            pos_weight: 类别不平衡权重，形状为 [num_classes] 或 None。
            smoothing: 平滑系数，范围为 [0, 1]。
        """
        super(WeightedCrossEntropyWithSmoothing, self).__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的预测输出 (logits)，形状为 [batch_size, num_classes]。
            targets: 真实标签，形状为 [batch_size, num_classes]。
        Returns:
            loss: 平滑后的 BCEWithLogitsLoss。
        """
        # 对目标标签进行平滑处理
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing * 0.5

        # 使用 BCEWithLogitsLoss 计算损失
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = criterion(inputs, targets_smooth)
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss as described in the paper."""
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature scaling parameter.
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        Args:
            features: Tensor of shape [batch_size, feature_dim], normalized embeddings.
            labels: Tensor of shape [batch_size], ground truth labels.
        Returns:
            loss: SupCon loss.
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize the features
        features = F.normalize(features, dim=1)  # Normalize embeddings
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]

        # Scale similarity matrix with temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create mask for positive pairs
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)  # Reshape labels to [batch_size, 1]
            mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size, batch_size]
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # Identity matrix for unsupervised

        # Compute log-softmax over similarity matrix
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)  # [batch_size, 1]
        logits = similarity_matrix - logits_max.detach()  # Stability adjustment
        exp_logits = torch.exp(logits)  # [batch_size, batch_size]
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))  # [batch_size, batch_size]

        # Only keep positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)  # [batch_size]

        # Compute loss
        loss = -mean_log_prob_pos.mean()  # Average over batch
        return loss


class SupCon_WCE_Loss(nn.Module):
    def __init__(self, weight=None, temperature=0.07, lambda1=1.0, lambda2=0.1):
        """
        Combined Loss: Weighted Cross Entropy Loss + Supervised Contrastive Loss
        Args:
            weight: 权重，用于 Weighted Cross Entropy Loss。
            temperature: SupConLoss 的温度参数。
            lambda1: Weighted Cross Entropy Loss 的权重系数。
            lambda2: SupConLoss 的权重系数。
        """
        super(SupCon_WCE_Loss, self).__init__()
        self.weighted_ce_loss = nn.CrossEntropyLoss(weight=weight)  # 加权交叉熵
        self.supcon_loss = SupConLoss(temperature=temperature)  # SupConLoss
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, logits, features, labels):
        """
        Args:
            logits: 模型的分类输出，形状为 [batch_size, num_classes]。
            features: 模型的特征输出，形状为 [batch_size, feature_dim]。
            labels: 样本的真实标签，形状为 [batch_size]。
        Returns:
            loss: 加权损失。
        """
        # 计算 Weighted Cross Entropy Loss
        ce_loss = self.weighted_ce_loss(logits, labels)

        # 计算 SupConLoss
        supcon_loss = self.supcon_loss(features, labels)

        # 加权组合
        total_loss = self.lambda1 * ce_loss + self.lambda2 * supcon_loss
        return total_loss
