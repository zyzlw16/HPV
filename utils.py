import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score, confusion_matrix
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, filepath)


def calculate_metrics(true_labels, pred_probs, threshold=0.5):
    # 转换 pred_probs 为 NumPy 数组
    pred_probs = np.array(pred_probs)
    pred_labels = (pred_probs >= threshold).astype(int)

    # 计算主要指标
    auc = roc_auc_score(true_labels, pred_probs)
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    ppv = precision

    return {
        "ROC-AUC": auc,
        "PR-AUC": pr_auc,
        "F1": f1,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "NPV": npv,
        "PPV": ppv
    }

def save_metrics(train_losses, val_losses, val_aucs, filepath):
    with open(filepath, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aucs": val_aucs
        }, f)

def plot_curves(train_losses, val_losses, val_aucs, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # 绘制 Loss 和 AUC 曲线
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", linestyle="-")
    plt.plot(epochs, val_losses, label="Val Loss", color="orange", linestyle="--")
    plt.plot(epochs, val_aucs, label="Val AUC", color="green", linestyle="-.")

    # 设置图例、标题和坐标轴
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Loss and AUC Curves")

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



