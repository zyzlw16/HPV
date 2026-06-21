import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
from src.FusionModel import FusionModel
from src.utils import save_checkpoint, calculate_metrics, plot_curves
import numpy as np

from src.train import load_config, setup_device, setup_loss_function, log_results, save_results
from src.datasets import BalancedSampler
from src.DualRegionDataLoader import DualRegionDataLoader


def setup_dataloaders(config, num_workers=4):
    """设置数据加载器"""
    augmentations = config.get("augmentations", {
        "flip": True,
        "rotate": True,
        "transpose": True
    })

    # 训练集
    train_dataset = DualRegionDataLoader(
        json_path=os.path.join(config["data_path"], f"fold{config['fold']}_data.json"),
        data_key="train",
        augmentations=augmentations
    )

    # 验证集
    val_dataset = DualRegionDataLoader(
        json_path=os.path.join(config["data_path"], f"fold{config['fold']}_data.json"),
        data_key="val",
        augmentations=None
    )

    # 数据加载器
    if config.get("sampler") == "balanced":
        train_sampler = BalancedSampler([item["label"] for item in train_dataset.data])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, config = None):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    train_ids = []

    for gtvp_input, gtvn_input, labels, ids in train_loader:
        # 将数据移到设备
        gtvp_input = gtvp_input.to(device)
        gtvn_input = gtvn_input.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()

        if config["loss_name"] == "SupCon_WCE":
            logits, features = model(gtvp_input, gtvn_input)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
            # 计算损失
            loss = criterion(logits, features, labels)
        else:
            logits, _ = model(gtvp_input, gtvn_input)

            # 🔧 修正1: 确保logits和labels维度正确
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
            loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        train_ids.extend(ids)

    # 计算指标
    avg_loss = running_loss / len(train_loader)

    return avg_loss, all_labels, all_probs,train_ids


def validate_epoch(model, val_loader, criterion, device, config = None):
    """验证一个epoch"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    val_ids = []

    with torch.no_grad():
        for gtvp_input, gtvn_input, labels, ids in val_loader:
            # 将数据移到设备
            gtvp_input = gtvp_input.to(device)
            gtvn_input = gtvn_input.to(device)
            labels = labels.to(device)

            if config["loss_name"] == "SupCon_WCE":
                logits, features = model(gtvp_input, gtvn_input)
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()
                # 计算损失
                loss = criterion(logits, features, labels)
            else:
                # 前向传播
                logits,_ = model(gtvp_input, gtvn_input)

                # 🔧 修正2: 确保logits和labels维度正确
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()

                loss = criterion(logits, labels)



            # 统计
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            val_ids.extend(ids)

    # 计算指标
    avg_loss = running_loss / len(val_loader)

    return avg_loss, all_labels, all_probs, val_ids


def train_fusion_model(config_path):
    """训练融合模型的主函数"""
    # 加载配置
    config = load_config(config_path)

    # 设置设备
    device = setup_device(config)

    #  创建保存目录
    os.makedirs(config["save_path"], exist_ok=True)

    # 创建日志文件
    log_file_path = os.path.join(config["save_path"], f"training_log_fold{config['fold']}.txt")
    log_file = open(log_file_path, "w")
    log_results(log_file, "Starting training...")
    log_results(log_file, f"Configuration: {json.dumps(config, indent=4)}")

    # 设置数据加载器
    train_loader, val_loader = setup_dataloaders(config)

    # 创建模型时使用新的参数名
    print("Creating fusion model...")
    model = FusionModel(
        gtvp_model_name=config.get("gtvp_model_name", "resnet18"),
        gtvn_model_name=config.get("gtvn_model_name", "resnet18"),
        gtvp_model_path=config.get("gtvp_model_path"),
        gtvn_model_path=config.get("gtvn_model_path"),
        freeze_gtvp=config.get("freeze_gtvp", True),
        freeze_gtvn=config.get("freeze_gtvn", True),
        fusion_type=config.get("fusion_type", "attention"),
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.5),
        unified_dim=config.get("unified_dim", 512),  # 🔥 新增参数
        input_size=config.get("input_size", (2, 100, 100, 100)),  # 🔥 新增参数
        num_attention_heads=config.get("num_attention_heads", 8)  # 🔥 新增参数
    ).to(device)


    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # 定义损失函数
    criterion = setup_loss_function(config, device, train_loader.dataset)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数
        lr=config["learning_rate"],
        weight_decay = config["weight_decay"]
    )

    # 学习率调度器监控验证损失（mode='min'）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控损失，越小越好
        patience=5,
        factor=0.5
    )

    # 训练循环
    best_loss = float("inf")
    best_acc = 0.0
    best_auc = 0.0  #  添加最佳AUC追踪
    train_losses = []
    val_losses = []
    val_aucs = []
    
    results_loss = {"train": [], "val": []}
    results_auc = {"train": [], "val": []}
    results_acc = {"train": [], "val": []}

    for epoch in range(config["max_epochs"]):
        log_results(log_file, f"\nEpoch {epoch + 1}/{config['max_epochs']}")

        # 训练
        train_loss, train_true, train_pred,train_ids = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )

        # 验证
        val_loss, val_true, val_pred, val_ids = validate_epoch(
            model, val_loader, criterion, device, config
        )

        # 计算训练集AUC
        train_pred_array = np.array(train_pred)
        train_auc = roc_auc_score(train_true, train_pred_array)

        #  计算验证集AUC（val_pred已经是概率值）
        val_pred_array = np.array(val_pred)        
        if val_pred_array.ndim == 2:
            val_pred_array = val_pred_array[:, 1]
        
        val_auc = roc_auc_score(val_true, val_pred_array)

        # 计算二分类预测和其他指标
        val_pred_binary = (val_pred_array >= 0.5).astype(int)
        val_f1 = f1_score(val_true, val_pred_binary)
        val_acc = accuracy_score(val_true, val_pred_binary)

        # 计算混淆矩阵
        cm = confusion_matrix(val_true, val_pred_binary)
        if cm.size == 4:  # 二分类
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = 0

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # 打印和记录结果
        log_msg = (
            f"Epoch {epoch + 1}/{config['max_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
            f"Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, "
            f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}"
        )

        log_results(log_file, log_msg)

        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        

        # 保存最佳模型-最小loss
        if isinstance(train_pred, list):
                train_pred = np.array(train_pred)
        if train_pred.ndim == 2:
                train_pred = train_pred[:, 1]
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/loss_best_model_fold{config['fold']}.pth")
            results_loss["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred_array}            
            results_loss["train"] = {"ID": train_ids,"true": train_true, "pred": train_pred}
        
        # 保存最佳模型-最大auc        
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/auc_best_model_fold{config['fold']}.pth")
            results_auc["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred_array}            
            results_auc["train"] = {"ID": train_ids,"true": train_true, "pred": train_pred}

        # 保存最佳模型-最大acc
        tn, fp, fn, tp = confusion_matrix(val_true, val_pred_binary).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (sensitivity + specificity) / 2
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/acc_best_model_fold{config['fold']}.pth")
            results_acc["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred_array}            
            results_acc["train"] = {"ID": train_ids, "true": train_true, "pred": train_pred}




    # 🔧 修正13: 训练结束后保存结果
    log_results(log_file, "\n" + "=" * 60)
    log_results(log_file, "Training completed!")
    log_results(log_file, f"Best Val Loss: {best_loss:.4f}")
    log_results(log_file, f"Best Val AUC: {best_auc:.4f}")
    log_results(log_file, "=" * 60)

    # 保存训练曲线
    try:
        plot_curves(
            train_losses,
            val_losses,
            val_aucs,
            save_path=os.path.join(config["save_path"], f"training_curves_fold{config['fold']}.png")
        )
        print(f" Saved training curves")
    except Exception as e:
        print(f" Failed to save training curves: {e}")

        
    # 保存结果
    save_results(results_loss, config,mode = 'loss')
    save_results(results_auc, config,mode = 'auc')
    save_results(results_acc, config,mode = 'acc')

    log_results(log_file, "Training complete.")
    log_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fusion Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    args = parser.parse_args()

    train_fusion_model(args.config)
