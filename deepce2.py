import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder  # 改用LeaveOneOutEncoder替换TargetEncoder
import joblib
import random
from sklearn.metrics import r2_score

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
# 改进的数据集类
# ======================
class EnhancedCarbonDataset(Dataset):
    def __init__(self, dataframe, target_col, emission_cols, cat_cols, num_cols, preprocessor=None):
        self.features = dataframe.drop(columns=emission_cols + [target_col])
        self.target = dataframe[target_col].copy()
        
        if preprocessor is None:
            # 分类特征处理 Pipeline（使用LeaveOneOutEncoder降低过拟合风险）
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LeaveOneOutEncoder(sigma=0.05))
            ])
            # 数值特征处理 Pipeline
            num_pipeline = Pipeline([
                ('imputer', IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    random_state=42)),
                ('power_transform', PowerTransformer()),
                ('scaler', RobustScaler())
            ])
            self.preprocessor = ColumnTransformer([
                ('cat', cat_pipeline, cat_cols),
                ('num', num_pipeline, num_cols)
            ])
            self.X = self.preprocessor.fit_transform(self.features, self.target)
        else:
            self.preprocessor = preprocessor
            self.X = self.preprocessor.transform(self.features)
        
        self.y_emissions = dataframe[emission_cols].values.astype(np.float32)
        self.y_total = dataframe[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.X[idx]),
            'emissions': torch.FloatTensor(self.y_emissions[idx]),
            'total': torch.FloatTensor(self.y_total[idx])
        }

# ======================
# 增强的模型架构（加入残差连接及更多正则化）
# ======================
class EnhancedMultiTaskModel(nn.Module):
    def __init__(self, input_size, num_emission_tasks, hidden_dims=[256, 128]):
        super().__init__()
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
            nn.Dropout(0.3)  # 增加Dropout比例
        )
        # 分项排放预测头（增加BatchNorm和Dropout）
        self.emission_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[1], 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Softplus()
            ) for _ in range(num_emission_tasks)
        ])
        # 总量预测头（包含跳过连接，并增加更多层和正则化）
        self.total_head = nn.Sequential(
            nn.Linear(hidden_dims[1] + input_size, 64),
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
        x = self.shared_block1(x)
        x = self.shared_block2(x)
        emissions = torch.cat([head(x) for head in self.emission_heads], dim=1)
        total_feat = torch.cat([x, identity], dim=1) 
        total = self.total_head(total_feat)
        return emissions, total

# ======================
# 改进的物理约束损失函数
# ======================
class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=0.7, temp=1.0):
        super().__init__()
        self.alpha = alpha    # 固定权重参数，必要时可设置为可学习参数
        self.temp = temp      # 温度系数
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        pred_emissions, pred_total = outputs
        true_emissions, true_total = targets
        
        emission_loss = self.mse(pred_emissions, true_emissions)
        total_loss = self.mse(pred_total, true_total)
        
        consistency_loss = torch.mean(
            torch.abs(pred_total - torch.sum(pred_emissions, dim=1, keepdim=True)) 
            / (torch.abs(pred_total.detach()) + self.temp)
        )
        
        return (1 - self.alpha) * emission_loss + self.alpha * total_loss + consistency_loss

# ======================
# 优化的训练流程（采用OneCycleLR学习率调度器以增强泛化能力）
# ======================
def train_enhanced_model(config):
    df = pd.read_excel(config['data_path'])
    target_col = 'Total_Carbon_Emissions'
    emission_cols = [c for c in df.columns if c.endswith('_Carbon_Emissions') and c != target_col]
    cat_cols = ['Battery_Type', 'Component_Country', 'Assembly_Country']
    num_cols = [c for c in df.columns if c not in cat_cols + emission_cols + [target_col]]
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = EnhancedCarbonDataset(train_df, target_col, emission_cols, cat_cols, num_cols)
    val_dataset = EnhancedCarbonDataset(val_df, target_col, emission_cols, cat_cols, num_cols, 
                                        train_dataset.preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']*2, 
                            shuffle=False, pin_memory=True)
    
    model = EnhancedMultiTaskModel(
        input_size=train_dataset.X.shape[1],
        num_emission_tasks=len(emission_cols),
        hidden_dims=config['hidden_dims']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 分层学习率优化器
    optimizer = optim.AdamW([
        {'params': model.shared_block1.parameters(), 'lr': config['lr']},
        {'params': model.shared_block2.parameters(), 'lr': config['lr']},
        {'params': model.emission_heads.parameters(), 'lr': config['lr'] * 1.5},
        {'params': model.total_head.parameters(), 'lr': config['lr'] * 2}
    ], weight_decay=config['weight_decay'])
    
    # 使用OneCycleLR调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr'] * 2, 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs']
    )
    criterion = PhysicsConstrainedLoss(alpha=0.7, temp=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    best_val_loss = float('inf')
    current_patience = config['patience']
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch['features'].to(device, non_blocking=True)
            emissions = batch['emissions'].to(device, non_blocking=True)
            total = batch['total'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                pred_emissions, pred_total = model(features)
                loss = criterion((pred_emissions, pred_total), (emissions, total))
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        val_r2_total = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device, non_blocking=True)
                emissions = batch['emissions'].to(device, non_blocking=True)
                total = batch['total'].to(device, non_blocking=True)
                
                pred_emissions, pred_total = model(features)
                loss = criterion((pred_emissions, pred_total), (emissions, total))
                val_loss += loss.item()
                
                val_r2_total.append(r2_score(
                    total.cpu().numpy(), 
                    pred_total.cpu().numpy()
                ))
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2 = np.mean(val_r2_total)
        
        if avg_val_loss < best_val_loss * 0.999:
            best_val_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'best_model.pth')
            current_patience = config['patience']
        else:
            current_patience -= 1
            if current_patience <= 0:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | R2 Total: {avg_val_r2:.4f}")
    
    joblib.dump(train_dataset.preprocessor, 'preprocessor.pth')
    print("Training completed.")

# ======================
# 预测接口（可选地在预测阶段引入MC Dropout进行不确定性估计）
# ======================
def predict_total_emission(input_data: dict) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    preprocessor = joblib.load('preprocessor.pth')
    feature_names = joblib.load('feature_names.pth') 
    emission_cols = joblib.load('emission_cols.pth')
    target_col = 'Total_Carbon_Emissions'
    cat_cols = ['Battery_Type', 'Component_Country', 'Assembly_Country']
    num_cols = [c for c in feature_names if c not in cat_cols]
    
    model = EnhancedMultiTaskModel(
        input_size=len(feature_names),
        num_emission_tasks=len(emission_cols),
        hidden_dims=config['hidden_dims']
    )
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    full_input = {feat: input_data.get(feat, np.nan) for feat in feature_names}
    df_input = pd.DataFrame([full_input])
    
    # 对缺失的分类和数值特征分别进行处理
    cat_data = df_input[cat_cols]
    num_data = df_input[num_cols]
    cat_imputer = preprocessor.named_transformers_['cat'].named_steps['imputer']
    num_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
    filled_cat = cat_imputer.transform(cat_data)
    filled_num = num_imputer.transform(num_data)
    
    filled_cat_df = pd.DataFrame(filled_cat, columns=cat_cols)
    filled_num_df = pd.DataFrame(filled_num, columns=num_cols)
    filled_df = pd.concat([filled_cat_df, filled_num_df], axis=1)
    print("\n填补后的特征值：")
    print(filled_df)
    
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat = encoder.transform(filled_cat)
    power_transform = preprocessor.named_transformers_['num'].named_steps['power_transform']
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    transformed_num = power_transform.transform(filled_num)
    scaled_num = scaler.transform(transformed_num)
    
    X_processed = np.hstack([encoded_cat, scaled_num])
    
    model.eval()
    with torch.no_grad():
        tensor_input = torch.FloatTensor(X_processed).to(device)
        pred_emissions, total_emission = model(tensor_input)

    total_value = total_emission.item()
    emission_values = pred_emissions.cpu().numpy().tolist()[0]

    return {"total_emission": total_value, "emission_breakdown": emission_values}

if __name__ == "__main__":
    config = {
        'data_path': '工作簿2.xlsx',
        'batch_size': 32,
        'lr': 7e-4,
        'min_lr': 1e-5,  # 保留备用
        'weight_decay': 2e-4,
        'epochs': 500,
        'patience': 30,
        'hidden_dims': [256, 192]
    }
    
    df = pd.read_excel(config['data_path'])
    emission_cols = [c for c in df.columns if c.endswith('_Carbon_Emissions') and c != 'Total_Carbon_Emissions']
    feature_names = df.drop(columns=emission_cols + ['Total_Carbon_Emissions']).columns.tolist()
    
    train_enhanced_model(config)
    
    joblib.dump(feature_names, 'feature_names.pth')
    joblib.dump(emission_cols, 'emission_cols.pth')
    
    sample_input = {
        'Battery_Type': 'LFP',
        'Mixing_anode': 1.5125,
        'Dry_room': 6.798389494,
        'Assembly_Country': 'China'
        # 其他特征缺失时将自动填补
    }
       
    prediction = predict_total_emission(sample_input)
    print(f"\n【预测结果】总碳排放量：{prediction['total_emission']:.2f} kgCO₂e")
