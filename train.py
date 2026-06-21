import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from src.datasets import HPVDataLoader, BalancedSampler, DualRegionDataLoader
from src.models import get_model
from src.losses import get_loss
from src.utils import save_checkpoint, calculate_metrics, plot_curves
import numpy as np
from src.WarmupCosine import WarmupCosineAnnealingLR  # 自定义的调度器


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_device(config):
    """设置设备"""
    return torch.device(config["device"] if torch.cuda.is_available() else "cpu")


def setup_dataloaders(config, num_workers=0):
    """设置数据加载器"""
    augmentations = config.get("augmentations", {"flip": True, "rotate": True, "transpose": True})

    # 根据data_mode和bootstrap参数拼接文件名
    fold = config["fold"]
    bootstrap = config.get("bootstrap", None)
    data_mode = config.get("data_mode", "plain")  # "plain" or "bootstrap"

    if data_mode == "bootstrap":
        if bootstrap is None:
            raise ValueError("在bootstrap模式下，config需指定bootstrap编号！")
        json_file = f"{config['data_path']}/fold{fold}_bootstrap{bootstrap}_data.json"
    else:
        json_file = f"{config['data_path']}/fold{fold}_data.json"


    if config["model_name"] == "dual_region":
        train_dataset = DualRegionDataLoader(
            json_path=json_file,
            data_key="train",
            augmentations=augmentations
        )
        val_dataset = DualRegionDataLoader(
            json_path=json_file,
            data_key="val",
            augmentations=None
        )
    else:
        train_dataset = HPVDataLoader(json_file, data_key="train", augmentations=augmentations)
        val_dataset = HPVDataLoader(json_file, data_key="val", augmentations=None)

    if config["sampler"] == "balanced":
        train_sampler = BalancedSampler([item["label"] for item in train_dataset.data])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def setup_model(config, device):
    """设置模型"""
    if config["model_name"] == "dual_region":
        model = get_model(
            name=config["model_name"],
            gtvp_model_path=config["gtvp_model_path"],
            gtvn_model_path=config["gtvn_model_path"],
            freeze_gtvp=config.get("freeze_gtvp", True),
            freeze_gtvn=config.get("freeze_gtvn", True),
            num_classes=config.get("num_classes", 2)
        ).to(device)
    else:
        model = get_model(
            name=config["model_name"],
            input_channels=2,
            #num_classes=config.get("num_classes", 2)
        ).to(device)
    #model = get_model(config["model_name"], input_channels=2).to(device)
    return model


def setup_loss_function(config, device, train_dataset=None):
    """设置损失函数"""
    if config["loss_name"] == "SupCon_WCE":

        labels = [item["label"] for item in train_dataset.data]
        pos_samples = sum(labels)
        neg_samples = len(labels) - pos_samples
        if pos_samples == 0 or neg_samples == 0:
            raise ValueError("The training dataset must contain both positive and negative samples.")
        num_samples = torch.tensor([neg_samples, pos_samples])  # 700 个类别 0 样本，300 个类别 1 样本
        class_weights = 1.0 / num_samples.float()

        # 获取增强因子，默认值为1
        neg_weight_factor = config.get("neg_weight_factor", 1)
        # 根据增强因子动态调整负样本权重
        class_weights[0] = class_weights[0] * neg_weight_factor  # 增强负样本权重
        class_weights = class_weights / class_weights.sum()  # 归一化
        loss_fn = get_loss(config["loss_name"], weight=class_weights,temperature=config.get("temperature", 0.07), lambda1=config.get("lambda_wce", 1), lambda2=config.get("lambda_supcon", 0.02)).to(device)
    else:
        if config["loss_name"] == "weighted_cross_entropy":
            labels = [item["label"] for item in train_dataset.data]
            pos_samples = sum(labels)
            neg_samples = len(labels) - pos_samples
            if pos_samples == 0 or neg_samples == 0:
                raise ValueError("The training dataset must contain both positive and negative samples.")
            num_samples = torch.tensor([neg_samples, pos_samples])  # 700 个类别 0 样本，300 个类别 1 样本
            class_weights = 1.0 / num_samples.float()

            # 获取增强因子，默认值为1
            neg_weight_factor = config.get("neg_weight_factor", 1)
            # 根据增强因子动态调整负样本权重
            class_weights[0] = class_weights[0] * neg_weight_factor  # 增强负样本权重
            class_weights = class_weights / class_weights.sum()  # 归一化
            loss_fn = get_loss(config["loss_name"], weight=class_weights).to(device)

        elif config["loss_name"] == "focal_loss":
            alpha = config.get("alpha", 0.25)
            gamma = config.get("gamma", 2.0)
            loss_fn = get_loss(config["loss_name"], alpha=alpha, gamma=gamma).to(device)
        else:
            loss_fn = get_loss(config["loss_name"]).to(device)

    return loss_fn

def setup_valid_loss_function(config, device, train_dataset=None):
    labels = [item["label"] for item in train_dataset.data]
    pos_samples = sum(labels)
    neg_samples = len(labels) - pos_samples
    if pos_samples == 0 or neg_samples == 0:
        raise ValueError("The training dataset must contain both positive and negative samples.")
    num_samples = torch.tensor([neg_samples, pos_samples])  # 700 个类别 0 样本，300 个类别 1 样本
    class_weights = 1.0 / num_samples.float()

    # 获取增强因子，默认值为1
    neg_weight_factor = config.get("neg_weight_factor", 1)
    # 根据增强因子动态调整负样本权重
    class_weights[0] = class_weights[0] * neg_weight_factor  # 增强负样本权重
    class_weights = class_weights / class_weights.sum()  # 归一化
    loss_fn = get_loss(config["loss_name"], weight=class_weights).to(device)
    return loss_fn



def train_one_epoch(model, train_loader, optimizer, device, loss_fn, config=None):
    """训练一个 epoch"""
    model.train()

    train_loss = 0.0
    train_true = []
    train_pred = []
    train_ids = []

    for inputs, labels, ids in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if config is not None and config["loss_name"] == "SupCon_WCE":
            # 前向传播
            outputs, features = model(inputs)
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
            # 计算损失
            loss = loss_fn(outputs, features, labels)

        else:
            outputs,_ = model(inputs)
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()


        train_loss += loss.item() * inputs.size(0)
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())
        train_ids.extend(ids)

    train_loss /= len(train_loader.dataset)
    return train_loss, train_true, train_pred,train_ids


def validate(model, val_loader, device, loss_fn,config=None):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    val_true = []
    val_pred = []
    val_ids = []

    with torch.no_grad():
        for inputs, labels, ids in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if config is not None and config["loss_name"] == "SupCon_WCE":
                outputs, features = model(inputs)
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()
                # 计算损失
                loss = loss_fn(outputs, features, labels)

            else:
                outputs,_ = model(inputs)
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()
                loss = loss_fn(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            val_ids.extend(ids)

    val_loss /= len(val_loader.dataset)
    return val_loss, val_true, val_pred, val_ids


def log_results(log_file, message):
    """日志记录模块"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def save_results(results, config,mode = None):

    """保存训练和验证结果"""
    # 保存训练集的预测值、真实值和数据 ID
    train_predictions = [
        {"id": id_, "true_label": int(true), "pred_score": float(pred)}
        for id_, true, pred in zip(results["train"]["ID"], results["train"]["true"], results["train"]["pred"])
    ]
    with open(config["save_path"] + f"/" + mode + f"_train_predictions_fold{config['fold']}.json", "w") as f:
        json.dump(train_predictions, f, indent=4)

    # 保存验证集的预测值、真实值和数据 ID
    val_predictions = [
        {"id": id_, "true_label": int(true), "pred_score": float(pred)}
        for id_, true, pred in zip(results["val"]["ID"], results["val"]["true"], results["val"]["pred"])
    ]
    with open(config["save_path"] + f"/" + mode + f"_val_predictions_fold{config['fold']}.json", "w") as f:
        json.dump(val_predictions, f, indent=4)

    # 计算指标并保存
    train_metrics = calculate_metrics(results["train"]["true"], results["train"]["pred"])
    val_metrics = calculate_metrics(results["val"]["true"], results["val"]["pred"])
    metrics = {"train": train_metrics, "val": val_metrics}
    with open(config["save_path"] + f"/" + mode + f"_metrics_fold{config['fold']}.json", "w") as f:
        json.dump(metrics, f, indent=4)


def train_model(config_path):
    """主训练函数"""
    # 加载配置和设备
    config = load_config(config_path)
    device = setup_device(config)

    # 创建日志文件
    log_file_path = os.path.join(config["save_path"], f"training_log_fold{config['fold']}.txt")
    log_file = open(log_file_path, "w")
    log_results(log_file, "Starting training...")
    log_results(log_file, f"Configuration: {json.dumps(config, indent=4)}")

    # 初始化数据加载器、模型和损失函数
    train_loader, val_loader = setup_dataloaders(config)
    model = setup_model(config, device)


    loss_fn = setup_loss_function(config, device, train_loader.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],weight_decay=config["weight_decay"])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float("inf")
    best_auc = 0
    best_acc = 0


    train_losses = []
    val_losses = []
    val_aucs = []
    results_loss = {"train": [], "val": []}
    results_auc = {"train": [], "val": []}
    results_acc = {"train": [], "val": []}

    # 判断 config 是否包含 checkpoint_path 参数
    if "checkpoint_path" in config and config["checkpoint_path"]:
        checkpoint_path = config["checkpoint_path"]
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型参数
            print(f"Checkpoint loaded successfully from {checkpoint_path}.")
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist. Starting training from scratch.")
    else:
        print("No checkpoint_path provided in config. Starting training from scratch.")

    for epoch in range(config["max_epochs"]):
        log_results(log_file, f"Epoch {epoch + 1}/{config['max_epochs']}")

        # 训练阶段
        if config["loss_name"] == "SupCon_WCE":
            train_loss, train_true, train_pred, train_ids = train_one_epoch(model, train_loader, optimizer, device, loss_fn, config)
        else:
            train_loss, train_true, train_pred, train_ids = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # 验证阶段
        if config["loss_name"] == "SupCon_WCE":
            val_loss, val_true, val_pred, val_ids = validate(model, val_loader, device, loss_fn, config)
        else:
            val_loss, val_true, val_pred, val_ids = validate(model, val_loader, device, loss_fn)

        

        # 如果是列表，先转换为 NumPy 数组
        if isinstance(val_pred, list):
            val_pred = np.array(val_pred)
        val_pred = val_pred[:, 1]

        # 保存最佳模型-最小loss
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/loss_best_model_fold{config['fold']}.pth")
            results_loss["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred}
            if isinstance(train_pred, list):
                train_pred = np.array(train_pred)
            results_loss["train"] = {"ID": train_ids,"true": train_true, "pred": train_pred[:, 1]}

        # 保存最佳模型-最大auc
        val_auc = roc_auc_score(val_true, val_pred)
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/auc_best_model_fold{config['fold']}.pth")
            results_auc["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred}
            if isinstance(train_pred, list):
                train_pred = np.array(train_pred)
            results_auc["train"] = {"ID": train_ids,"true": train_true, "pred": train_pred[:, 1]}

        val_pred_binary = [1 if p >= 0.5 else 0 for p in val_pred]
        f1 = f1_score(val_true, val_pred_binary)
        tn, fp, fn, tp = confusion_matrix(val_true, val_pred_binary).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (sensitivity + specificity) / 2
        # 保存最佳模型-最大acc
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            save_checkpoint(model, optimizer, epoch, config["save_path"] + f"/acc_best_model_fold{config['fold']}.pth")
            results_acc["val"] = {"ID": val_ids, "true": val_true, "pred": val_pred}
            if isinstance(train_pred, list):
                train_pred = np.array(train_pred)
            results_acc["train"] = {"ID": train_ids, "true": train_true, "pred": train_pred[:, 1]}

        # 更新学习率
        scheduler.step(val_loss)
        
        val_aucs.append(val_auc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_results(log_file, f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
                              f"F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, "
                              f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    # 保存训练曲线
    plot_curves(train_losses, val_losses, val_aucs, config["save_path"] + f"/training_curves_fold{config['fold']}.png")

    # 保存结果
    save_results(results_loss, config,mode = 'loss')
    save_results(results_auc, config,mode = 'auc')
    save_results(results_acc, config,mode = 'acc')

    log_results(log_file, "Training complete.")
    log_file.close()
