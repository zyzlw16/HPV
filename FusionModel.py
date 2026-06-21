import torch
import torch.nn as nn
from src.models import get_model

class FusionModel(nn.Module):
    """
    特征融合模型，支持多种融合方式：
    - cat: 拼接融合（先投影+BN，再拼接）
    - attention: 交叉注意力融合（GTVp作为Query，GTVn作为Key/Value，生成新特征后与GTVp拼接）
    - weighted: 动态自适应加权融合（每个患者使用不同的权重参数）
    """

    def __init__(
            self,
            gtvp_model_name="resnet18",
            gtvn_model_name="resnet18",
            gtvp_model_path=None,
            gtvn_model_path=None,
            freeze_gtvp=True,
            freeze_gtvn=True,
            fusion_type="attention",
            num_classes=2,
            dropout=0.5,
            unified_dim=512,
            input_size=(2, 100, 100, 100),  # 🔥 支持 (C, D, H, W) 格式
            num_attention_heads=8,  # 注意力头数
    ):
        """
        Args:
            gtvp_model_name: GTVp模型名称
            gtvn_model_name: GTVn模型名称
            gtvp_model_path: GTVp预训练模型路径
            gtvn_model_path: GTVn预训练模型路径
            freeze_gtvp: 是否冻结GTVp模型参数
            freeze_gtvn: 是否冻结GTVn模型参数
            fusion_type: 融合方式 ("cat", "attention", "weighted")
            num_classes: 分类类别数
            dropout: Dropout比率
            unified_dim: 统一投影维度
            input_size: 输入尺寸 (C, D, H, W)
            num_attention_heads: 注意力头数（仅用于attention融合）
        """
        super(FusionModel, self).__init__()

        self.fusion_type = fusion_type
        self.unified_dim = unified_dim
        self.num_classes = num_classes

        # ========== 1. 加载预训练模型 ==========
        print("=" * 60)
        print("🔧 初始化融合模型")
        print("=" * 60)

        # 加载GTVp模型
        print("\n📥 加载 GTVp 模型...")
        self.gtvp_model = get_model(gtvp_model_name, num_classes=num_classes)
        if gtvp_model_path:
            self._load_pretrained_weights(self.gtvp_model, gtvp_model_path)

        # 加载GTVn模型
        print("\n📥 加载 GTVn 模型...")
        self.gtvn_model = get_model(gtvn_model_name, num_classes=num_classes)
        if gtvn_model_path:
            self._load_pretrained_weights(self.gtvn_model, gtvn_model_path)

        # 冻结或解冻模型参数
        self._set_requires_grad(self.gtvp_model, not freeze_gtvp)
        self._set_requires_grad(self.gtvn_model, not freeze_gtvn)
        print(f"  → GTVp 参数冻结: {freeze_gtvp}")
        print(f"  → GTVn 参数冻结: {freeze_gtvn}")

        # ========== 2. 移除分类头，保留特征提取部分 ==========
        print("\n🔨 移除分类头...")
        self.gtvp_features = self._remove_classifier(self.gtvp_model)
        self.gtvn_features = self._remove_classifier(self.gtvn_model)

        # ========== 3. 自动计算特征维度 ==========
        print("\n🔍 自动检测特征维度...")
        feature_dim_gtvp = self._get_feature_dim(self.gtvp_features, input_size)
        feature_dim_gtvn = self._get_feature_dim(self.gtvn_features, input_size)

        print(f"\n✅ GTVp 原始特征维度: {feature_dim_gtvp}")
        print(f"✅ GTVn 原始特征维度: {feature_dim_gtvn}")
        print(f"✅ 统一投影维度: {unified_dim}")


        # ========== 4. 定义投影层 + BatchNorm（融合前的预处理）==========
        print(f"\n🎯 构建融合架构 (类型: {fusion_type})...")

        # GTVp特征投影到统一维度 + BatchNorm
        self.gtvp_projection = nn.Sequential(
            nn.Linear(feature_dim_gtvp, unified_dim),
            nn.BatchNorm1d(unified_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # GTVn特征投影到统一维度 + BatchNorm
        self.gtvn_projection = nn.Sequential(
            nn.Linear(feature_dim_gtvn, unified_dim),
            nn.BatchNorm1d(unified_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 初始化知识分解模块
        self.knowledge_decomposition = Knowledge_Decomposition(
            feat_len=unified_dim,
            feat_dim=unified_dim
        )
        print(f"✅ 知识分解模块已初始化 (feat_len={unified_dim}, feat_dim={unified_dim})")



        # ========== 5. 定义融合层 ==========
        if fusion_type == "cat":
            # 🔹 拼接融合：直接拼接两个对齐后的特征
            fusion_dim = unified_dim * 2
            self.fusion_fc = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.BatchNorm1d(fusion_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, num_classes)
            )
            print(f"  → 拼接融合维度: {unified_dim} + {unified_dim} = {fusion_dim}")

        elif fusion_type in ["attention", "attention_transformer"]:
            # 🔹 交叉注意力融合：GTVp作为Query，GTVn作为Key/Value
            # 生成的注意力特征再与GTVp拼接
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=unified_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )

            # 注意力输出的LayerNorm
            self.attention_norm = nn.LayerNorm(unified_dim)

            # 拼接后的融合层（GTVp + 注意力特征）
            fusion_dim = unified_dim * 2
            self.fusion_fc = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.BatchNorm1d(fusion_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, num_classes)
            )
            print(f"  → 交叉注意力融合:")
            print(f"    • Query: GTVp ({unified_dim})")
            print(f"    • Key/Value: GTVn ({unified_dim})")
            print(f"    • 注意力头数: {num_attention_heads}")
            print(f"    • 最终拼接维度: {fusion_dim}")

            self.transformer = Transformer(feature_dim=unified_dim * 2)  # 拼接后维度是 2倍

        elif fusion_type == "knowledge_decomposition":
            # self.fusion_fc = nn.Sequential(
            #     nn.Linear(unified_dim * 4, unified_dim * 2),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(unified_dim * 2, num_classes)
            # )
            self.fusion_fc = nn.Sequential(
                nn.Linear(unified_dim * 4, unified_dim * 2),
                nn.BatchNorm1d(unified_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(unified_dim * 2, num_classes)
            )
            # 初始化 Transformer 模块
            self.transformer = Transformer(feature_dim=unified_dim * 4)  # 拼接后维度是 4 倍

        elif fusion_type == "weighted":
            # 🔹 动态自适应加权融合：每个患者使用不同的权重
            # 使用小型网络生成患者特异性权重
            self.weight_generator = nn.Sequential(
                nn.Linear(unified_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 2),  # 生成2个权重（GTVp和GTVn）
                nn.Softmax(dim=1)  # 归一化为概率分布
            )

            self.fusion_fc = nn.Sequential(
                nn.Linear(unified_dim, unified_dim // 2),
                nn.BatchNorm1d(unified_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(unified_dim // 2, num_classes)
            )
            print(f"  → 动态加权融合:")
            print(f"    • 权重生成器: 自适应学习每个患者的权重")
            print(f"    • 加权特征维度: {unified_dim}")
        else:
            raise ValueError(f"❌ 不支持的融合类型: {fusion_type}")

        print("\n" + "=" * 60)
        print("✅ 模型初始化完成！")
        print("=" * 60 + "\n")

    def _load_pretrained_weights(self, model, model_path):
        """加载预训练权重"""
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  ✓ 加载预训练权重: {model_path}")

    def _set_requires_grad(self, model, requires_grad):
        """设置模型参数是否需要梯度"""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _remove_classifier(self, model):
        """移除模型的分类头"""
        # 检查是否是 TinyResNet3D 模型
        if hasattr(model, 'fc1') and hasattr(model, 'fc2'):
            print(f"  ✓ 检测到 TinyResNet3D 模型")
            print(f"    - fc1: {model.fc1}")
            print(f"    - fc2: {model.fc2}")

            # 移除分类头，替换为 Identity
            model.fc1 = nn.Identity()
            model.fc2 = nn.Identity()
            model.dropout1 = nn.Identity()
            model.dropout2 = nn.Identity()
            print(f"  ✓ 已移除分类头 (fc1, fc2, dropout1, dropout2)")
            return model

        # 如果是其他模型，尝试通用方法
        elif hasattr(model, 'fc'):
            print(f"  ✓ 检测到标准 ResNet 模型")
            model.fc = nn.Identity()
            print(f"  ✓ 已移除分类头 (fc)")
            return model

        else:
            print(f"  ⚠ 警告：未能识别模型结构，请手动检查")
            return model

    def _get_feature_dim(self, model, input_size):
        """
        🔥 自动计算模型的输出特征维度（支持2D/3D/5D输入）

        Args:
            model: 特征提取模型
            input_size: 输入尺寸
                - 2D: (C, H, W) 例如 (3, 224, 224)
                - 3D: (C, D, H, W) 例如 (2, 100, 100, 100)

        Returns:
            feature_dim: 特征维度
        """
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)

        print(f"  → Dummy输入形状: {dummy_input.shape}")

        model.eval()
        with torch.no_grad():
            try:
                output,_ = model(dummy_input)
            except Exception as e:
                raise RuntimeError(
                    f"❌ 模型前向传播失败！\n"
                    f"输入形状: {dummy_input.shape}\n"
                    f"错误信息: {str(e)}"
                )

        # print(f"  → 模型输出形状: {output.shape}")

        # 处理不同维度的输出
        if len(output.shape) == 2:
            feature_dim = output.shape[1]
        elif len(output.shape) == 3:
            feature_dim = output.view(output.size(0), -1).shape[1]
        elif len(output.shape) == 4:
            feature_dim = output.view(output.size(0), -1).shape[1]
        elif len(output.shape) == 5:
            feature_dim = output.view(output.size(0), -1).shape[1]
        else:
            raise ValueError(f"❌ 不支持的输出维度: {output.shape}")

        print(f"  → 展平后特征维度: {feature_dim}")
        return feature_dim

    def forward(self, x_gtvp, x_gtvn):
        """
        前向传播

        Args:
            x_gtvp: GTVp模型的输入 (batch_size, C, D, H, W)
            x_gtvn: GTVn模型的输入 (batch_size, C, D, H, W)

        Returns:
            logits: 分类logits (batch_size, num_classes)
            attention_weights: 注意力权重（仅在attention模式下返回，可选）
        """
        batch_size = x_gtvp.size(0)

        # ========== 1. 提取原始特征 ==========
        feat_gtvp = self.gtvp_features(x_gtvp)
        feat_gtvn = self.gtvn_features(x_gtvn)

        # 检查是否为元组，并提取 features
        if isinstance(feat_gtvp, tuple):
            feat_gtvp = feat_gtvp[1]
        if isinstance(feat_gtvn, tuple):
            feat_gtvn = feat_gtvn[1]

        # ========== 2. 展平特征（如果是多维的）==========
        if len(feat_gtvp.shape) > 2:
            feat_gtvp = feat_gtvp.view(batch_size, -1)
        if len(feat_gtvn.shape) > 2:
            feat_gtvn = feat_gtvn.view(batch_size, -1)

        # ========== 3. 投影到统一维度 + BatchNorm ==========
        feat_gtvp_aligned = self.gtvp_projection(feat_gtvp)  # (B, unified_dim)
        feat_gtvn_aligned = self.gtvn_projection(feat_gtvn)  # (B, unified_dim)

        # ========== 4. 特征融合 ==========
        if self.fusion_type == "cat":
            # 🔹 方式1: 直接拼接
            fused_features = torch.cat([feat_gtvp_aligned, feat_gtvn_aligned], dim=1)
            logits = self.fusion_fc(fused_features)
            return logits, fused_features

        elif self.fusion_type == "attention":
            # 🔹 方式2: 交叉注意力融合
            # GTVp作为Query，GTVn作为Key和Value
            # 需要添加序列维度 (B, 1, D)
            query = feat_gtvp_aligned.unsqueeze(1)  # (B, 1, unified_dim)
            key_value = feat_gtvn_aligned.unsqueeze(1)  # (B, 1, unified_dim)

            # 交叉注意力
            attn_output, attn_weights = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=True
            )

            # 移除序列维度并归一化
            attn_output = attn_output.squeeze(1)  # (B, unified_dim)
            attn_output = self.attention_norm(attn_output)

            # 将注意力特征与GTVp特征拼接
            fused_features = torch.cat([feat_gtvp_aligned, attn_output], dim=1)  # (B, unified_dim*2)

            logits = self.fusion_fc(fused_features)

            # 可选：返回注意力权重用于可视化
            # return logits, attn_weights
            return logits, fused_features

        elif self.fusion_type == "attention_transformer":
            # 🔹 方式2: 交叉注意力融合
            # GTVp作为Query，GTVn作为Key和Value
            # 需要添加序列维度 (B, 1, D)
            query = feat_gtvp_aligned.unsqueeze(1)  # (B, 1, unified_dim)
            key_value = feat_gtvn_aligned.unsqueeze(1)  # (B, 1, unified_dim)

            # 交叉注意力
            attn_output, attn_weights = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=True
            )

            # 移除序列维度并归一化
            attn_output = attn_output.squeeze(1)  # (B, unified_dim)
            attn_output = self.attention_norm(attn_output)

            # 将注意力特征与GTVp特征拼接
            fused_features = torch.cat([feat_gtvp_aligned, attn_output], dim=1)  # (B, unified_dim*2)

            fused_features, _ = self.transformer(fused_features)
            logits = self.fusion_fc(fused_features)

            # 可选：返回注意力权重用于可视化
            # return logits, attn_weights
            return logits, fused_features

        elif self.fusion_type == "knowledge_decomposition":
            common, synergy, gtvp_spec, gtvn_spec = self.knowledge_decomposition(
                feat_gtvp_aligned, feat_gtvn_aligned)

            # 5. 拼接特征
            indiv_know = torch.cat([common, synergy, gtvp_spec, gtvn_spec], dim=1)  # (B, unified_dim * 4)

            # 6. Transformer 建模
            fusion, _ = self.transformer(indiv_know)

            # 7. 分类器输出
            logits = self.fusion_fc(fusion)
            return logits, fusion


        elif self.fusion_type == "weighted":
            # 🔹 方式3: 动态自适应加权融合
            # 拼接特征用于生成权重
            concat_features = torch.cat([feat_gtvp_aligned, feat_gtvn_aligned], dim=1)

            # 为每个患者生成自适应权重
            weights = self.weight_generator(concat_features)  # (B, 2)
            weight_gtvp = weights[:, 0:1]  # (B, 1)
            weight_gtvn = weights[:, 1:2]  # (B, 1)

            # 动态加权融合
            fused_features = weight_gtvp * feat_gtvp_aligned + weight_gtvn * feat_gtvn_aligned

            logits = self.fusion_fc(fused_features)
            return logits, fused_features

        else:
            raise ValueError(f"❌ 不支持的融合类型: {self.fusion_type}")


class Specificity_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )

    def forward(self, feat):
        feat = self.conv(feat)
        return feat


class Interaction_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
        super().__init__()
        self.gtvp_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        self.gtvn_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        self.gtvp_atten = nn.Linear(dim, 1)
        self.gtvn_atten = nn.Linear(dim, 1)

    def forward(self, gtvp_feat, gtvn_feat):
        """
        输入:
        - gtvp_feat: Tensor(batch_size, feature_dim) -> (6, 512)
        - gtvn_feat: Tensor(batch_size, feature_dim) -> (6, 512)

        输出:
        - interaction: Tensor(batch_size, feature_dim) -> (6, 512)
        """
        # 特征变换
        gtvp_align = self.gtvp_fc(gtvp_feat)  # (batch_size, feature_dim)
        gtvn_align = self.gtvn_fc(gtvn_feat)  # (batch_size, feature_dim)

        # 扩展维度以计算交互注意力
        # gtvp_align.unsqueeze(1): (batch_size, 1, feature_dim)
        # gtvn_align.unsqueeze(0): (1, batch_size, feature_dim)
        atten = gtvp_align.unsqueeze(1) * gtvn_align.unsqueeze(0)  # (batch_size, batch_size, feature_dim)

        # 计算注意力权重
        gtvp_att = torch.sigmoid(self.gtvp_atten(atten))  # (batch_size, batch_size, 1)
        gtvn_att = torch.sigmoid(self.gtvn_atten(atten.permute(1, 0, 2)))  # (batch_size, batch_size, 1)

        # 去掉最后一维，得到注意力矩阵
        gtvp_att = gtvp_att.squeeze(-1)  # (batch_size, batch_size)
        gtvn_att = gtvn_att.squeeze(-1)  # (batch_size, batch_size)

        # 加权特征
        gtvp_weighted = torch.matmul(gtvp_att, gtvp_align)  # (batch_size, feature_dim)
        gtvn_weighted = torch.matmul(gtvn_att, gtvn_align)  # (batch_size, feature_dim)

        # 交互特征
        interaction = gtvp_weighted + gtvn_weighted  # (batch_size, feature_dim)

        return interaction


class Knowledge_Decomposition(nn.Module):
    def __init__(self, feat_len=6, feat_dim=64):
        super().__init__()
        self.gtvp_spec = Specificity_Estimator(feat_len, feat_dim)
        self.gtvn_spec = Specificity_Estimator(feat_len, feat_dim)

        self.common_encoder = Interaction_Estimator(feat_len, feat_dim)
        self.synergy_encoder = Interaction_Estimator(feat_len, feat_dim)

    def forward(self, gtvp_feat, gtvn_feat):
        gtvp_spec = self.gtvp_spec(gtvp_feat)
        gtvn_spec = self.gtvn_spec(gtvn_feat)
        common = self.common_encoder(gtvn_feat, gtvp_feat)
        synergy = self.synergy_encoder(gtvn_feat, gtvp_feat)
        return common, synergy, gtvp_spec, gtvn_spec

class Transformer(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.layer2 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        B = features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        features = features.unsqueeze(1)  # 将 [batch_size, feature_dim] 转为 [batch_size, 1, feature_dim]
        h = torch.cat((cls_tokens, features), dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

# ========== 测试代码 ==========
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 测试融合模型")
    print("=" * 60 + "\n")


    # 模拟简单的3D模型
    class Simple3DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(2, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1)
            )
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


    # 测试三种融合方式
    fusion_types = ["cat", "attention", "weighted"]

    for fusion_type in fusion_types:
        print(f"\n{'=' * 60}")
        print(f"测试融合类型: {fusion_type.upper()}")
        print(f"{'=' * 60}\n")

        # 创建模型
        model = FusionModel(
            gtvp_model_name="resnet18",
            gtvn_model_name="resnet18",
            fusion_type=fusion_type,
            num_classes=3,
            unified_dim=256,
            input_size=(2, 100, 100, 100),
            dropout=0.3,
            num_attention_heads=8
        )

        # 创建测试数据
        batch_size = 4
        x_gtvp = torch.randn(batch_size, 2, 100, 100, 100)
        x_gtvn = torch.randn(batch_size, 2, 100, 100, 100)

        print(f"\n📊 测试前向传播:")
        print(f"  → GTVp 输入形状: {x_gtvp.shape}")
        print(f"  → GTVn 输入形状: {x_gtvn.shape}")

        # 前向传播
        output = model(x_gtvp, x_gtvn)

        print(f"\n✅ 输出形状: {output.shape}")
        print(f"✅ 预期形状: ({batch_size}, 3)")

        assert output.shape == (batch_size, 3), f"❌ 输出形状不正确！"
        print(f"\n✅ {fusion_type.upper()} 融合测试通过！\n")

    print("=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
