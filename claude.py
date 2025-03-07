import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder
import joblib
import random
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# ----------------------
# 可重复性设置
# ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from sklearn import config_context
    with config_context(assume_finite=True, working_memory=128):
        np.random.seed(seed)
set_seed(42)

# ======================
# 改进的数据集类（强化阴极材料特征）
# ======================
class EnhancedCarbonDataset(Dataset):
    def __init__(self, dataframe, target_col, emission_cols, cat_cols, num_cols, 
                 cathode_cols=None, preprocessor=None):
        """
        增强的碳排放数据集，特别处理阴极材料特征
        
        参数:
            dataframe: 输入数据框
            target_col: 总碳排放列名
            emission_cols: 分项碳排放列名列表
            cat_cols: 分类特征列名列表
            num_cols: 数值特征列名列表
            cathode_cols: 阴极材料相关列名列表
            preprocessor: 预处理器（如果已有）
        """
        self.df_original = dataframe.copy()  # 保存原始数据框用于特殊处理
        self.cathode_cols = cathode_cols if cathode_cols else []
        
        # 寻找阴极材料相关列
        if not self.cathode_cols:
            self.cathode_cols = [col for col in dataframe.columns if 'Cathode' in col and col not in emission_cols + [target_col]]
            print(f"自动识别的阴极材料列: {self.cathode_cols}")
        
        self.features = dataframe.drop(columns=emission_cols + [target_col])
        self.target = dataframe[target_col].copy()
        
        # 保存特征列名供后续使用
        self.all_feature_names = self.features.columns.tolist()
        
        # 添加阴极材料的交互特征和多项式特征
        if self.cathode_cols:
            for col in self.cathode_cols:
                if col in self.features.columns:
                    # 添加二次项和三次项
                    self.features[f"{col}_squared"] = self.features[col] ** 2
                    self.features[f"{col}_cubed"] = self.features[col] ** 3
                    
                    # 与电池类型的交互
                    if 'Battery_Type' in self.features.columns:
                        battery_dummies = pd.get_dummies(self.features['Battery_Type'], prefix='BT')
                        for bt_col in battery_dummies.columns:
                            self.features[f"{col}_{bt_col}"] = self.features[col] * battery_dummies[bt_col]
            
            # 更新数值特征列表
            added_cols = [c for c in self.features.columns if c not in self.all_feature_names]
            num_cols = list(num_cols) + added_cols
            self.all_feature_names = self.features.columns.tolist()
            print(f"添加的新特征: {added_cols}")
            
        # 保存原始阴极特征值（用于直接物理约束）
        self.cathode_original = {}
        for col in self.cathode_cols:
            if col in self.features.columns:
                self.cathode_original[col] = self.features[col].values.astype(np.float32)
        
        if preprocessor is None:
            # 分类特征处理 Pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LeaveOneOutEncoder(sigma=0.05))
            ])
            
            # 阴极特征特殊处理 Pipeline（较轻的缩放）
            cathode_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler(with_centering=True, with_scaling=False))  # 只中心化不缩放
            ])
            
            # 其他数值特征处理 Pipeline
            other_num_cols = [col for col in num_cols if col not in self.cathode_cols and not any(col.startswith(c) for c in self.cathode_cols)]
            num_pipeline = Pipeline([
                ('imputer', IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    random_state=42)),
                ('power_transform', PowerTransformer()),
                ('scaler', RobustScaler())
            ])
            
            # 构建预处理器，根据是否有阴极特征决定处理方式
            transformers = [('cat', cat_pipeline, cat_cols)]
            
            if self.cathode_cols:
                cathode_feature_cols = self.cathode_cols + [col for col in added_cols if any(col.startswith(c) for c in self.cathode_cols)]
                transformers.append(('cathode', cathode_pipeline, cathode_feature_cols))
                
            if other_num_cols:
                transformers.append(('num', num_pipeline, other_num_cols))
                
            self.preprocessor = ColumnTransformer(transformers)
            self.X = self.preprocessor.fit_transform(self.features, self.target)
            
            # 保存特征名和重要特性
            self.feature_names_after_transform = self.get_feature_names()
            
            # 记录阴极特征的索引位置
            self.cathode_indices = []
            for col in self.cathode_cols:
                if col in self.all_feature_names:
                    try:
                        idx = self.all_feature_names.index(col)
                        trans_idx = self.get_transformed_index(idx)
                        if trans_idx is not None:
                            self.cathode_indices.append(trans_idx)
                    except:
                        print(f"无法找到列 {col} 的转换后索引")
                        
            print(f"阴极特征在转换后的索引: {self.cathode_indices}")
            
        else:
            self.preprocessor = preprocessor
            self.X = self.preprocessor.transform(self.features)
            self.feature_names_after_transform = preprocessor.feature_names_after_transform \
                if hasattr(preprocessor, 'feature_names_after_transform') else None
            self.cathode_indices = preprocessor.cathode_indices \
                if hasattr(preprocessor, 'cathode_indices') else []
        
        self.y_emissions = dataframe[emission_cols].values.astype(np.float32)
        self.y_total = dataframe[target_col].values.astype(np.float32).reshape(-1, 1)
        
        # 标识阴极相关的排放列
        self.cathode_emission_indices = []
        for i, col in enumerate(emission_cols):
            if any(c in col for c in ['Cathode', 'Material']):
                self.cathode_emission_indices.append(i)
        print(f"阴极相关排放列索引: {self.cathode_emission_indices}")

    def get_feature_names(self):
        """获取转换后的特征名称"""
        try:
            return self.preprocessor.get_feature_names_out()
        except:
            # 如果不支持get_feature_names_out，创建通用名称
            return [f"feature_{i}" for i in range(self.X.shape[1])]
    
    def get_transformed_index(self, original_idx):
        """获取原始特征在转换后的索引位置"""
        try:
            for name, trans, cols in self.preprocessor.transformers_:
                if self.all_feature_names[original_idx] in cols:
                    col_idx = list(cols).index(self.all_feature_names[original_idx])
                    # 计算在转换后特征中的位置
                    col_pos = sum([len(c) for _, _, c in self.preprocessor.transformers_ if _ < name])
                    return col_pos + col_idx
        except:
            pass
        return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'features': torch.FloatTensor(self.X[idx]),
            'emissions': torch.FloatTensor(self.y_emissions[idx]),
            'total': torch.FloatTensor(self.y_total[idx])
        }
        
        # 添加原始阴极特征值用于物理约束
        for col, values in self.cathode_original.items():
            sample[f'cathode_{col}'] = torch.FloatTensor([values[idx]])
            
        return sample

# ======================
# 深度增强的模型架构（专门处理阴极材料特征）
# ======================
class EnhancedMultiTaskModel(nn.Module):
    def __init__(self, input_size, num_emission_tasks, hidden_dims=[256, 128], 
                 cathode_indices=None, use_focal_attention=True):
        """
        增强的多任务模型，专门处理阴极材料特征
        
        参数:
            input_size: 输入特征维度
            num_emission_tasks: 排放任务数量
            hidden_dims: 隐藏层维度列表
            cathode_indices: 阴极特征在输入中的索引位置
            use_focal_attention: 是否使用焦点注意力机制
        """
        super().__init__()
        self.cathode_indices = cathode_indices if cathode_indices else []
        self.use_focal_attention = use_focal_attention
        
        # 共享层部分
        self.shared_block1 = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3)
        )
        self.shared_block2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.3)
        )
        
        # 阴极材料专门处理分支（如果有指定阴极特征索引）
        if self.cathode_indices:
            self.cathode_branch = nn.Sequential(
                nn.Linear(len(self.cathode_indices), 64),
                nn.GELU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            
            # 焦点注意力机制（使阴极特征对某些排放有更强影响）
            if use_focal_attention:
                self.cathode_attention = nn.Sequential(
                    nn.Linear(32, num_emission_tasks),
                    nn.Sigmoid()
                )
        
        # 分项排放预测头
        self.emission_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[1] + (32 if self.cathode_indices else 0), 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Softplus()
            ) for _ in range(num_emission_tasks)
        ])
        
        # 总量预测头（包含跳过连接和原始特征）
        self.total_head = nn.Sequential(
            nn.Linear(hidden_dims[1] + input_size + (32 if self.cathode_indices else 0), 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        identity = x
        
        # 提取阴极特征并处理（如果有）
        cathode_feat = None
        cathode_attention = None
        if self.cathode_indices:
            cathode_input = torch.stack([x[:, i] for i in self.cathode_indices], dim=1)
            cathode_feat = self.cathode_branch(cathode_input)
            
            # 生成注意力权重（如果启用）
            if self.use_focal_attention:
                cathode_attention = self.cathode_attention(cathode_feat)
        
        # 处理共享层
        x = self.shared_block1(x)
        x = self.shared_block2(x)
        
        # 生成分项排放预测
        emissions_list = []
        for i, head in enumerate(self.emission_heads):
            if cathode_feat is not None:
                # 应用阴极特征和注意力
                if self.use_focal_attention:
                    # 使用注意力机制调节阴极特征的影响
                    weighted_cathode = cathode_feat * cathode_attention[:, i].unsqueeze(1)
                    combined_feat = torch.cat([x, weighted_cathode], dim=1)
                else:
                    combined_feat = torch.cat([x, cathode_feat], dim=1)
                emissions_list.append(head(combined_feat))
            else:
                emissions_list.append(head(x))
        
        emissions = torch.cat(emissions_list, dim=1)
        
        # 总预测（包含原始特征和阴极特征）
        if cathode_feat is not None:
            total_feat = torch.cat([x, identity, cathode_feat], dim=1)
        else:
            total_feat = torch.cat([x, identity], dim=1)
            
        total = self.total_head(total_feat)
        
        return emissions, total

# ======================
# 物理导向的碳排放损失函数
# ======================
class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=0.7, temp=1.0, cathode_weight=3.0, emission_indices=None):
        """
        物理约束的损失函数，特别关注阴极材料的影响
        
        参数:
            alpha: 总排放损失的权重
            temp: 一致性损失的温度系数
            cathode_weight: 阴极材料约束的权重
            emission_indices: 阴极相关排放的索引
        """
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.cathode_weight = cathode_weight
        self.cathode_emission_indices = emission_indices if emission_indices else []
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, outputs, targets, cathode_values=None):
        pred_emissions, pred_total = outputs
        true_emissions, true_total = targets
        
        # 基本损失：分项排放和总排放的MSE
        emission_loss = self.mse(pred_emissions, true_emissions)
        total_loss = self.mse(pred_total, true_total)
        
        # 一致性损失：确保分项排放之和与总排放一致
        consistency_loss = torch.mean(
            torch.abs(pred_total - torch.sum(pred_emissions, dim=1, keepdim=True)) 
            / (torch.abs(pred_total.detach()) + self.temp)
        )
        
        # 物理约束损失：确保阴极材料用量与相关排放有直接关系
        physical_loss = 0.0
        if cathode_values is not None and self.cathode_emission_indices:
            for idx in self.cathode_emission_indices:
                cathode_emission = pred_emissions[:, idx].unsqueeze(1)
                
                # 比例关系约束：排放量应与材料用量近似成正比
                # 计算单位材料的排放量
                unit_emission = cathode_emission / (cathode_values + 1e-6)
                
                # 单位排放应该在样本间保持相对一致
                mean_unit_emission = torch.mean(unit_emission)
                variance_loss = torch.mean(
                    torch.square(unit_emission - mean_unit_emission) 
                    / (mean_unit_emission + 1e-6)
                )
                
                # 确保材料用量增加时排放也增加（梯度方向一致）
                gradient_loss = torch.mean(
                    torch.relu(-(cathode_emission[1:] - cathode_emission[:-1]) * 
                              (cathode_values[1:] - cathode_values[:-1]))
                )
                
                physical_loss += (variance_loss + gradient_loss) * 0.5
            
            physical_loss *= self.cathode_weight
        
        # 综合损失
        total_combined_loss = (1 - self.alpha) * emission_loss + self.alpha * total_loss + \
                             consistency_loss + physical_loss
        
        return total_combined_loss

# ======================
# 增强训练流程
# ======================
def train_enhanced_model(config):
    """增强的模型训练流程，特别关注阴极材料的影响"""
    df = pd.read_excel(config['data_path'])
    target_col = 'Total_Carbon_Emissions'
    emission_cols = [c for c in df.columns if c.endswith('_Carbon_Emissions') and c != target_col]
    cat_cols = ['Battery_Type', 'Component_Country', 'Assembly_Country']
    
    # 识别阴极材料相关列
    cathode_cols = [c for c in df.columns if 'Cathode' in c and c not in emission_cols + [target_col]]
    print(f"识别到的阴极材料列: {cathode_cols}")
    
    num_cols = [c for c in df.columns if c not in cat_cols + emission_cols + [target_col]]
    
    # 数据探索分析
    print("\n数据探索:")
    for col in cathode_cols:
        if col in df.columns:
            print(f"{col} 统计: 均值={df[col].mean():.4f}, 标准差={df[col].std():.4f}, 最小值={df[col].min():.4f}, 最大值={df[col].max():.4f}")
            
            # 检查阴极材料用量与总排放的相关性
            corr = df[col].corr(df[target_col])
            print(f"{col} 与 {target_col} 的相关性: {corr:.4f}")
            
            # 检查阴极材料用量与阴极相关排放的相关性
            for e_col in emission_cols:
                if 'Cathode' in e_col or 'Material' in e_col:
                    e_corr = df[col].corr(df[e_col])
                    print(f"{col} 与 {e_col} 的相关性: {e_corr:.4f}")
    
    # 创建训练和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 创建数据集
    train_dataset = EnhancedCarbonDataset(
        train_df, target_col, emission_cols, cat_cols, num_cols, cathode_cols
    )
    val_dataset = EnhancedCarbonDataset(
        val_df, target_col, emission_cols, cat_cols, num_cols, cathode_cols,
        preprocessor=train_dataset.preprocessor
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size']*2, 
        shuffle=False, pin_memory=True
    )
    
    # 初始化模型
    model = EnhancedMultiTaskModel(
        input_size=train_dataset.X.shape[1],
        num_emission_tasks=len(emission_cols),
        hidden_dims=config['hidden_dims'],
        cathode_indices=train_dataset.cathode_indices,
        use_focal_attention=True
    )
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 优化器配置（使用分层学习率）
    param_groups = [
        {'params': model.shared_block1.parameters(), 'lr': config['lr']},
        {'params': model.shared_block2.parameters(), 'lr': config['lr']}
    ]
    
    # 为阴极分支设置更高学习率
    if hasattr(model, 'cathode_branch'):
        param_groups.append({'params': model.cathode_branch.parameters(), 'lr': config['lr'] * 2})
        if hasattr(model, 'cathode_attention'):
            param_groups.append({'params': model.cathode_attention.parameters(), 'lr': config['lr'] * 2})
    
    # 为排放头设置中等学习率
    param_groups.append({'params': model.emission_heads.parameters(), 'lr': config['lr'] * 1.5})
    
    # 为总排放预测头设置较高学习率
    param_groups.append({'params': model.total_head.parameters(), 'lr': config['lr'] * 2})
    
    optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g['lr'] * 2 for g in param_groups], 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs'],
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    # 损失函数
    criterion = PhysicsConstrainedLoss(
        alpha=0.7, 
        temp=1.0, 
        cathode_weight=3.0,
        emission_indices=train_dataset.cathode_emission_indices
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    # 训练循环
    best_val_loss = float('inf')
    current_patience = config['patience']
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2_total': [],
        'val_r2_emissions': []
    }
    
    print("\n开始训练:")
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            # 提取批次数据
            features = batch['features'].to(device, non_blocking=True)
            emissions = batch['emissions'].to(device, non_blocking=True)
            total = batch['total'].to(device, non_blocking=True)
            
            # 提取阴极材料原始值（用于物理约束）
            cathode_values = None
            if train_dataset.cathode_cols:
                col = train_dataset.cathode_cols[0]  # 使用第一个阴极列作为示例
                if f'cathode_{col}' in batch:
                    cathode_values = batch[f'cathode_{col}'].to(device, non_blocking=True)
            
            # 前向传播（混合精度）
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                pred_emissions, pred_total = model(features)
                loss = criterion(
                    (pred_emissions, pred_total), 
                    (emissions, total),
                    cathode_values
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_r2_total = []
        val_r2_emissions = [[] for _ in range(len(emission_cols))]
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device, non_blocking=True)
                emissions = batch['emissions'].to(device, non_blocking=True)
                total = batch['total'].to(device, non_blocking=True)
                
                # 提取阴极材料原始值
                cathode_values = None
                if val_dataset.cathode_cols:
                    col = val_dataset.cathode_cols[0]
                    if f'cathode_{col}' in batch:
                        cathode_values = batch[f'cathode_{col}'].to(device, non_blocking=True)
                
                # 前向传播
                pred_emissions, pred_total = model(features)
                loss = criterion(
                    (pred_emissions, pred_total), 
                    (emissions, total),
                    cathode_values
                )
                val_loss += loss.item()
                
                # 计算R^2分数
                val_r2_total.append(r2_score(
                    total.cpu().numpy(), 
                    pred_total.cpu().numpy()
                ))
                
                # 计算每个排放项的R^2
                for i in range(len(emission_cols)):
                    val_r2_emissions[i].append(r2_score(
                        emissions[:, i].cpu().numpy(),
                        pred_emissions[:, i].cpu().numpy()
                    ))
        
        # 计算平均损失和R^2
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2_total = np.mean(val_r2_total)
        avg_val_r2_emissions = [np.mean(r2) for r2 in val_r2_emissions]
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_r2_total'].append(avg_val_r2_total)
        history['val_r2_emissions'].append(avg_val_r2_emissions)
        
        # 检查早停
        if avg_val_loss < best_val_loss * 0.999:
            best_val_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_r2_total': avg_val_r2_total,
                
                'cathode_indices': train_dataset.cathode_indices,
                'cathode_emission_indices': train_dataset.cathode_emission_indices
            }, 'best_model.pth')
            current_patience = config['patience']
            print(f"  [保存模型] 验证损失改善: {avg_val_loss:.4f}")
        else:
            current_patience -= 1
            if current_patience <= 0:
                print(f"[早停] 在 epoch {epoch + 1} 停止训练")
                break
        
        # 打印进度
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | R2 Total: {avg_val_r2_total:.4f}")
        
        # 每5个epoch打印一次详细R2
        if (epoch + 1) % 5 == 0:
            for i, col in enumerate(emission_cols):
                print(f"  - {col} R2: {avg_val_r2_emissions[i]:.4f}")
            
            # 对于阴极相关的排放项，打印更详细信息
            for i in train_dataset.cathode_emission_indices:
                print(f"  * 阴极相关: {emission_cols[i]} R2: {avg_val_r2_emissions[i]:.4f}")
    
    # 保存预处理器和其他重要信息
    model_info = {
        'preprocessor': train_dataset.preprocessor,
        'feature_names': train_dataset.all_feature_names,
        'emission_cols': emission_cols,
        'cathode_cols': train_dataset.cathode_cols,
        'cathode_indices': train_dataset.cathode_indices,
        'cathode_emission_indices': train_dataset.cathode_emission_indices,
        'target_col': target_col,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'input_size': train_dataset.X.shape[1],
        'hidden_dims': config['hidden_dims']
    }
    joblib.dump(model_info, 'model_info.pth')
    
    # 绘制训练历史
    plot_training_history(history, emission_cols)
    
    # 训练结束后进行特征重要性分析
    analyze_feature_importance(model, train_dataset, device)
    
    # 进行阴极材料的敏感性分析
    if train_dataset.cathode_cols:
        sensitivity_analysis_cathode(
            model, 
            train_dataset, 
            val_df, 
            train_dataset.cathode_cols[0], 
            device
        )
    
    print("训练完成。")
    return model, model_info

# ======================
# 辅助函数：绘制训练历史
# ======================
def plot_training_history(history, emission_cols):
    """绘制训练历史曲线"""
    # 创建输出目录
    os.makedirs('plots', exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/loss_history.png')
    plt.close()
    
    # 绘制R2曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_r2_total'], label='总排放 R2')
    plt.title('验证集 R2 分数')
    plt.xlabel('Epoch')
    plt.ylabel('R2 分数')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/r2_history.png')
    plt.close()
    
    # 绘制各排放项的R2曲线
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(emission_cols):
        r2_values = [epoch_r2[i] for epoch_r2 in history['val_r2_emissions']]
        plt.plot(r2_values, label=col.replace('_Carbon_Emissions', ''))
    plt.title('各排放项的验证集 R2 分数')
    plt.xlabel('Epoch')
    plt.ylabel('R2 分数')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/emissions_r2_history.png')
    plt.close()

# ======================
# 特征重要性分析
# ======================
def analyze_feature_importance(model, dataset, device):
    """分析特征重要性"""
    print("\n进行特征重要性分析...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 将特征和目标变量转换为numpy数组
    X = dataset.X
    y_total = dataset.y_total
    
    # 创建预测函数
    def predict_func(X_samples):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_samples).to(device)
            _, total_pred = model(X_tensor)
            return total_pred.cpu().numpy()
    
    # 计算排列重要性（这可能需要一些时间）
    print("计算特征重要性中，这可能需要几分钟...")
    result = permutation_importance(
        predict_func, X, y_total,
        n_repeats=5, random_state=42,
        n_jobs=-1 if device.type == 'cpu' else 1
    )
    
    # 获取特征名称
    if hasattr(dataset, 'feature_names_after_transform') and dataset.feature_names_after_transform is not None:
        feature_names = dataset.feature_names_after_transform
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    }).sort_values('Importance', ascending=False)
    
    # 打印结果
    print("\n特征重要性排序:")
    print(importance_df.head(10))
    
    # 检查阴极相关特征的重要性
    if dataset.cathode_indices:
        cathode_feature_names = [feature_names[i] for i in dataset.cathode_indices if i < len(feature_names)]
        print("\n阴极相关特征重要性:")
        for name in cathode_feature_names:
            if name in importance_df['Feature'].values:
                rank = importance_df[importance_df['Feature'] == name].index[0] + 1
                importance = importance_df[importance_df['Feature'] == name]['Importance'].values[0]
                print(f"  {name}: 重要性 = {importance:.4f}, 排名 = {rank}/{len(feature_names)}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'], xerr=top_features['Std'])
    plt.title('排列重要性（总排放）')
    plt.xlabel('重要性（均值减少）')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    return importance_df

# ======================
# 阴极材料敏感性分析
# ======================
def sensitivity_analysis_cathode(model, dataset, val_df, cathode_col, device, num_points=20):
    """对阴极材料进行敏感性分析"""
    print(f"\n对阴极材料列 '{cathode_col}' 进行敏感性分析...")
    
    if cathode_col not in val_df.columns:
        print(f"错误：在验证集中找不到列 '{cathode_col}'")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 获取阴极材料的值范围
    min_val = val_df[cathode_col].min()
    max_val = val_df[cathode_col].max()
    mean_val = val_df[cathode_col].mean()
    
    # 获取典型的样本行作为基准
    sample_row = val_df.iloc[0].copy()
    
    # 生成测试点
    test_values = np.linspace(min_val * 0.5, max_val * 1.5, num_points)
    
    # 收集结果
    total_emissions = []
    component_emissions = []
    
    # 对每个测试值进行预测
    for test_val in test_values:
        # 创建测试样本
        test_sample = sample_row.copy()
        test_sample[cathode_col] = test_val
        
        # 准备特征
        test_df = pd.DataFrame([test_sample])
        
        # 提取特征和目标
        features = test_df.drop(columns=dataset.df_original.columns[dataset.df_original.columns.str.endswith('_Carbon_Emissions')])
        
        # 应用预处理
        X_processed = dataset.preprocessor.transform(features)
        
        # 进行预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_processed).to(device)
            pred_emissions, pred_total = model(X_tensor)
            
            total_emissions.append(pred_total.item())
            component_emissions.append(pred_emissions.cpu().numpy()[0])
    
    # 转换为数组
    total_emissions = np.array(total_emissions)
    component_emissions = np.array(component_emissions)
    
    # 创建DataFrame便于分析
    results_df = pd.DataFrame({
        cathode_col: test_values,
        'Total_Emission': total_emissions
    })
    
    # 添加各组件排放
    emission_cols = dataset.df_original.columns[dataset.df_original.columns.str.endswith('_Carbon_Emissions')].tolist()
    for i, col in enumerate(emission_cols):
        if i < component_emissions.shape[1]:
            results_df[col] = component_emissions[:, i]
    
    # 计算变化敏感性
    baseline_idx = np.abs(test_values - mean_val).argmin()
    baseline_total = total_emissions[baseline_idx]
    
    results_df['Total_Change_Pct'] = ((results_df['Total_Emission'] - baseline_total) / baseline_total) * 100
    
    # 打印分析结果
    print("\n敏感性分析结果:")
    print(f"  阴极材料值范围: {min_val:.4f} 到 {max_val:.4f}, 均值: {mean_val:.4f}")
    print(f"  当 {cathode_col} 增加一倍时，总排放变化: {results_df.iloc[-1]['Total_Change_Pct']:.2f}%")
    
    # 找出对阴极材料最敏感的排放组件
    if dataset.cathode_emission_indices:
        print("\n阴极材料相关排放项变化:")
        for idx in dataset.cathode_emission_indices:
            if idx < len(emission_cols):
                col = emission_cols[idx]
                baseline_comp = component_emissions[baseline_idx, idx]
                max_comp = component_emissions[-1, idx]
                change_pct = ((max_comp - baseline_comp) / baseline_comp) * 100
                print(f"  {col}: 当材料增加一倍时变化 {change_pct:.2f}%")
    
    # 绘制敏感性曲线
    plt.figure(figsize=(12, 8))
    plt.plot(test_values, total_emissions, 'o-', linewidth=2, color='blue')
    plt.axvline(x=mean_val, color='gray', linestyle='--', label=f'平均值 ({mean_val:.2f})')
    plt.axvline(x=min_val, color='red', linestyle=':', label=f'最小值 ({min_val:.2f})')
    plt.axvline(x=max_val, color='green', linestyle=':', label=f'最大值 ({max_val:.2f})')
    plt.title(f'{cathode_col} 对总碳排放的敏感性分析')
    plt.xlabel(cathode_col)
    plt.ylabel('总碳排放')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/sensitivity_{cathode_col}_total.png')
    plt.close()
    
    # 绘制各组件排放敏感性曲线
    plt.figure(figsize=(14, 8))
    for idx in dataset.cathode_emission_indices:
        if idx < len(emission_cols):
            col = emission_cols[idx]
            plt.plot(test_values, component_emissions[:, idx], 'o-', linewidth=2, label=col.replace('_Carbon_Emissions', ''))
    
    plt.axvline(x=mean_val, color='gray', linestyle='--', label=f'平均值 ({mean_val:.2f})')
    plt.title(f'{cathode_col} 对各组件碳排放的敏感性分析')
    plt.xlabel(cathode_col)
    plt.ylabel('组件碳排放')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/sensitivity_{cathode_col}_components.png')
    plt.close()
    
    return results_df

# ======================
# 预测接口
# ======================
def predict_total_emission(input_data: dict, device=None) -> dict:
    """提供输入特征，预测总碳排放和各组件碳排放"""
    # 确定设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型信息
    model_info = joblib.load('model_info.pth')
    preprocessor = model_info['preprocessor']
    feature_names = model_info['feature_names']
    emission_cols = model_info['emission_cols']
    cathode_indices = model_info['cathode_indices']
    hidden_dims = model_info['hidden_dims']
    input_size = model_info['input_size']
    
    # 初始化模型
    model = EnhancedMultiTaskModel(
        input_size=input_size,
        num_emission_tasks=len(emission_cols),
        hidden_dims=hidden_dims,
        cathode_indices=cathode_indices,
        use_focal_attention=True
    )
    
    # 加载模型权重
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # 准备输入数据
    full_input = {feat: input_data.get(feat, np.nan) for feat in feature_names}
    df_input = pd.DataFrame([full_input])
    
    # 添加阴极材料多项式特征和交互特征
    cathode_cols = model_info['cathode_cols']
    if cathode_cols:
        for col in cathode_cols:
            if col in df_input.columns:
                # 添加二次项和三次项
                df_input[f"{col}_squared"] = df_input[col] ** 2
                df_input[f"{col}_cubed"] = df_input[col] ** 3
                
                # 与电池类型的交互
                if 'Battery_Type' in df_input.columns:
                    battery_type = df_input['Battery_Type'].iloc[0]
                    for bt in ['LFP', 'NMC', 'LCO']:  # 假设这些是可能的电池类型
                        col_name = f"{col}_BT_{bt}"
                        df_input[col_name] = df_input[col] * (1 if battery_type == bt else 0)
    
    # 应用预处理
    X_processed = preprocessor.transform(df_input)
    
    # 进行预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(device)
        pred_emissions, pred_total = model(X_tensor)
    
    # 获取预测结果
    total_value = pred_total.item()
    emission_values = pred_emissions.cpu().numpy().tolist()[0]
    
    # 构建结果字典
    emission_dict = {}
    for i, col in enumerate(emission_cols):
        short_name = col.replace('_Carbon_Emissions', '')
        emission_dict[short_name] = emission_values[i]
    
    result = {
        "total_emission": total_value,
        "emission_breakdown": emission_dict
    }
    
    # 打印原始输入和结果
    print("\n输入特征:")
    for key, value in input_data.items():
        print(f"  {key}: {value}")
        
    print("\n预测结果:")
    print(f"  总碳排放: {total_value:.4f} kgCO₂e")
    print("  组件排放明细:")
    for name, value in emission_dict.items():
        print(f"    {name}: {value:.4f} kgCO₂e")
    
    # 如果有阴极材料，进行额外的分析
    if cathode_cols and any(col in input_data for col in cathode_cols):
        for col in cathode_cols:
            if col in input_data:
                print(f"\n阴极材料 '{col}' 分析:")
                
                # 计算阴极材料对相关排放的贡献比例
                cathode_emission_indices = model_info['cathode_emission_indices']
                for idx in cathode_emission_indices:
                    if idx < len(emission_cols):
                        emission_col = emission_cols[idx]
                        short_name = emission_col.replace('_Carbon_Emissions', '')
                        emission_value = emission_values[idx]
                        percentage = (emission_value / total_value) * 100
                        print(f"  贡献 {short_name}: {emission_value:.4f} kgCO₂e ({percentage:.2f}% 总排放)")
    
    return result

# ======================
# 主函数
# ======================
if __name__ == "__main__":
    # 配置参数
    config = {
        'data_path': '工作簿2.xlsx',
        'batch_size': 32,
        'lr': 5e-4,            # 略微调高学习率
        'weight_decay': 1e-4,  # 降低权重衰减以减少正则化
        'epochs': 500,
        'patience': 40,        # 增加耐心值，允许模型更充分训练
        'hidden_dims': [256, 192]
    }
    
    # 训练模型
    model, model_info = train_enhanced_model(config)
    
    # 示例输入
    sample_input = {
        'Battery_Type': 'LFP',
        'Cathode_Material_Amount': 1.5,  # 阴极材料用量
        'Mixing_anode': 1.5125,
        'Dry_room': 6.798389494,
        'Assembly_Country': 'China'
        # 其他特征缺失时将自动填补
    }
    
    # 预测
    prediction = predict_total_emission(sample_input)
    print(f"\n【示例预测结果】总碳排放量：{prediction['total_emission']:.2f} kgCO₂e")
    
    # 进行阴极材料敏感性测试
    print("\n阴极材料敏感性测试:")
    test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for test_val in test_values:
        test_input = sample_input.copy()
        test_input['Cathode_Material_Amount'] = test_val
        pred = predict_total_emission(test_input)
        print(f"  阴极材料 = {test_val}: 总排放 = {pred['total_emission']:.4f} kgCO₂e")