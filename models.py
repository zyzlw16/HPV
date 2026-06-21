import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Components for 3D ResNet ---

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock3D(nn.Module):
    """
    A basic block for a 3D ResNet, consisting of two 3D convolutions.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# --- Model Definitions ---

def get_model(name, **kwargs):
    """
    Model factory.
    """
    if name == "resnet18":
        return ResNet18Model(**kwargs)
    elif name == "dual_region":
        return DualRegionFusionModel(
            gtvp_model_path=kwargs["gtvp_model_path"],
            gtvn_model_path=kwargs["gtvn_model_path"],
            freeze_gtvp=kwargs.get("freeze_gtvp", True),
            freeze_gtvn=kwargs.get("freeze_gtvn", True),
            num_classes=kwargs.get("num_classes", 2)
        )
    elif name == "fusion":
        return FusionModel(**kwargs)
    elif name == "tiny_resnet":  # 添加 TinyResNet3D 到工厂函数
        return TinyResNet3D(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")


class ResNet18Model(nn.Module):
    """
    A from-scratch implementation of ResNet-18 for 3D image data.
    """

    def __init__(self, input_channels=2, num_classes=2, dropout_rate=0.5, return_features=False):
        super(ResNet18Model, self).__init__()
        self.return_features = return_features  # 新增：控制是否返回特征

        self.inplanes = 64
        # Initial convolution layer for 3D input
        self.conv1 = nn.Conv3d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(BasicBlock3D, 64, 2)
        self.layer2 = self._make_layer(BasicBlock3D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, 512, 2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  # 修正：bias 而不是 batchias

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 保存特征

        if self.return_features:
            return features  # 返回特征用于融合模型

        # 正常分类流程
        x = self.dropout(features)
        x = self.fc(x)
        return x


class TinyResNet3D(nn.Module):
    """针对小数据集的轻量级3D ResNet"""

    def __init__(self, input_channels=2, num_classes=2, dropout_rate=0.5):  # 添加 input_channels 参数
        super(TinyResNet3D, self).__init__()
        self.expansion = BasicBlock3D.expansion

        # 超参数 - 大幅减少通道数
        self.in_channels = 16  # 起始通道数从64减少到16
        channels = [16, 32, 64, 128]  # 各层通道数

        # 初始卷积层 - 使用较小的kernel和stride
        self.conv1 = nn.Conv3d(input_channels, self.in_channels, kernel_size=3,  # 使用 input_channels
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)

        # 残差层 - 只有4层，而不是标准的4层x[2,2,2,2]=16层
        self.layer1 = self._make_layer(BasicBlock3D, channels[0], 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, channels[1], 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, channels[2], 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, channels[3], 2, stride=2)

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类头 - 添加强正则化
        self.dropout1 = nn.Dropout3d(dropout_rate)
        self.fc1 = nn.Linear(channels[3] * self.expansion, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 池化和分类
        x = self.avgpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        features = torch.flatten(x, 1)  # 提取特征

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits  = self.fc2(x)

        return logits, features  # 返回分类结果和特征


# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FusionModel(nn.Module):
    """
    Original FusionModel. Note: This class may have a design issue.
    """

    def __init__(self, input_channels=2):
        super(FusionModel, self).__init__()
        self.resnet_gtvp = ResNet18Model(input_channels)
        self.resnet_gtvn = ResNet18Model(input_channels)
        self.fc_fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_gtvp, x_gtvn):
        features_gtvp = self.resnet_gtvp(x_gtvp)
        features_gtvn = self.resnet_gtvn(x_gtvn)
        fused_features = torch.cat((features_gtvp, features_gtvn), dim=1)
        output = self.fc_fusion(fused_features)
        return output


class ProjectionHead(nn.Module):
    """
    投影头，用于生成对比学习的嵌入。
    """

    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class DualRegionFusionModel(nn.Module):
    """
    双区域特征融合模型 - 修正版本
    """

    def __init__(self, gtvp_model_path, gtvn_model_path, freeze_gtvp=True, freeze_gtvn=True, num_classes=2):
        super(DualRegionFusionModel, self).__init__()

        # 加载 GTVp 分支 - 设置为返回特征
        self.gtvp_branch = ResNet18Model(input_channels=2, num_classes=num_classes, return_features=True)
        checkpoint = torch.load(gtvp_model_path)
        self.gtvp_branch.load_state_dict(checkpoint["model_state_dict"])
        if freeze_gtvp:
            for param in self.gtvp_branch.parameters():
                param.requires_grad = False

        # 加载 GTVn 分支 - 设置为返回特征
        self.gtvn_branch = ResNet18Model(input_channels=2, num_classes=num_classes, return_features=True)
        checkpoint = torch.load(gtvn_model_path)
        self.gtvn_branch.load_state_dict(checkpoint["model_state_dict"])
        if freeze_gtvn:
            for param in self.gtvn_branch.parameters():
                param.requires_grad = False

        # 自适应加权融合
        self.fusion_weights = nn.Parameter(torch.Tensor([0.5, 0.5]))

        # Batch Normalization
        self.bn_gtvp = nn.BatchNorm1d(512)
        self.bn_gtvn = nn.BatchNorm1d(512)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 修正：连接两个512维特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, gtvp_input, gtvn_input):
        # 获取特征而不是分类结果
        gtvp_features = self.gtvp_branch(gtvp_input)
        gtvn_features = self.gtvn_branch(gtvn_input)

        # Batch Normalization
        gtvp_features = self.bn_gtvp(gtvp_features)
        gtvn_features = self.bn_gtvn(gtvn_features)

        # 连接特征而不是加权融合
        fused_features = torch.cat([gtvp_features, gtvn_features], dim=1)

        output = self.classifier(fused_features)
        return output