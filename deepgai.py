import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder
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
set_seed(42)

# ======================
# 动态特征智能处理
# ======================
class AutoFeatureEngineer:
    def __init__(self):
        self.feature_mappings = {
            'EnergyDensity': ('Battery_Voltage', 'Battery_Weight'),
            'ProductionEfficiency': ('Material_Used', 'Production_Time')
        }
        self.stats = {}

    def validate_features(self, df):
        """智能特征验证与创建"""
        available_features = {}
        for feat, req_cols in self.feature_mappings.items():
            if all(col in df.columns for col in req_cols):
                available_features[feat] = req_cols
        return available_features

    def create_features(self, df):
        """自动化特征生成"""
        created_features = []
        if 'EnergyDensity' in self.feature_mappings:
            voltage_col, weight_col = self.feature_mappings['EnergyDensity']
            if voltage_col in df.columns and weight_col in df.columns:
                df['Energy_Density'] = df[voltage_col] / (df[weight_col] + 1e-6)
                df[weight_col] = df[weight_col].clip(upper=df[weight_col].quantile(0.95))
                created_features.append('Energy_Density')
        
        if 'ProductionEfficiency' in self.feature_mappings:
            material_col, time_col = self.feature_mappings['ProductionEfficiency']
            if material_col in df.columns and time_col in df.columns:
                df['Production_Efficiency'] = df[material_col] / (df[time_col] + 1e-6)
                created_features.append('Production_Efficiency')
        
        for col in df.select_dtypes(include=np.number):
            self.stats[col] = {
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75)
            }
            df[col] = df[col].clip(self.stats[col]['q1'], self.stats[col]['q3'])
        return df, created_features

# ======================
# 改进的数据集类
# ======================
class CarbonEmissionDataset(Dataset):
    def __init__(self, dataframe, target_col, config, preprocessor=None):
        self.afe = AutoFeatureEngineer()
        self.df_processed, new_features = self.afe.create_features(dataframe.copy())
        self.target_col = target_col
        
        # 动态识别列类型
        self.cat_cols = [col for col in config['possible_cat_cols'] if col in self.df_processed.columns]
        self.num_cols = [col for col in self.df_processed.select_dtypes(include=np.number).columns 
                        if col not in self.cat_cols + [target_col]]
        self.emission_cols = [col for col in self.df_processed.columns 
                             if col.endswith('Emission') and col != target_col]

        # 预处理管道
        if preprocessor is None:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LeaveOneOutEncoder(sigma=0.05))
            ])
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            self.preprocessor = ColumnTransformer([
                ('cat', cat_pipeline, self.cat_cols),
                ('num', num_pipeline, self.num_cols)
            ])
            self.X = self.preprocessor.fit_transform(self.df_processed)
        else:
            self.preprocessor = preprocessor
            self.X = self.preprocessor.transform(self.df_processed)

        self.y_total = self.df_processed[target_col].values.astype(np.float32).reshape(-1, 1)
        if self.emission_cols:
            self.y_emissions = self.df_processed[self.emission_cols].values.astype(np.float32)
        else:
            self.y_emissions = np.zeros((len(self.df_processed), 1), dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.X[idx]),
            'emissions': torch.FloatTensor(self.y_emissions[idx]),
            'total': torch.FloatTensor(self.y_total[idx])
        }

# ======================
# 鲁棒的多任务模型架构
# ======================
class MultiTaskEmissionsModel(nn.Module):
    def __init__(self, input_size, num_tasks, hidden_dims=[256, 128]):
        super().__init__()
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.SiLU()
        )
        # 动态任务头生成
        self.emission_heads = nn.ModuleList([
            self._make_head(hidden_dims[-1]) for _ in range(num_tasks)
        ])
        self.total_head = self._make_head(hidden_dims[-1], output_dim=1)

    def _make_head(self, input_dim, output_dim=1):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        shared_out = self.shared(x)
        emissions = torch.cat([head(shared_out) for head in self.emission_heads], dim=1)
        total = self.total_head(shared_out)
        return emissions, total

# ======================
# 强化约束的损失函数
# ======================
class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, temp=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.temp = temp

    def forward(self, outputs, targets):
        pred_emissions, pred_total = outputs
        true_emissions, true_total = targets
        
        # 基础回归损失
        emission_loss = self.mse(pred_emissions, true_emissions)
        total_loss = self.mse(pred_total, true_total)
        
        # 物理一致性约束
        sum_emissions = torch.sum(pred_emissions, dim=1, keepdim=True)
        consistency_loss = torch.mean(
            torch.abs(pred_total - sum_emissions) / 
            (torch.abs(pred_total.detach()) + self.temp)
        )
        
        return self.alpha*emission_loss + (1-self.alpha)*total_loss + self.beta*consistency_loss

# ======================
# 智能训练系统
# ======================
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self):
        df = pd.read_excel(self.config['data_path'])
        self.target_col = 'Total_Carbon_Emissions'
        self.afe = AutoFeatureEngineer()
        self.afe.create_features(df)
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_dataset = CarbonEmissionDataset(
            train_df, self.target_col, self.config
        )
        self.val_dataset = CarbonEmissionDataset(
            val_df, self.target_col, self.config, 
            self.train_dataset.preprocessor
        )
        print(f"训练样本数: {len(self.train_dataset)}, 验证样本数: {len(self.val_dataset)}")
    
    def build_model(self):
        num_tasks = len(self.train_dataset.emission_cols)
        self.model = MultiTaskEmissionsModel(
            input_size=self.train_dataset.X.shape[1],
            num_tasks=num_tasks,
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=8, factor=0.5
        )
        self.criterion = CompositeLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in DataLoader(self.train_dataset, 
                              batch_size=self.config['batch_size'], 
                              shuffle=True):
            self.optimizer.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs['features'])
                loss = self.criterion(outputs, (inputs['emissions'], inputs['total']))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_dataset)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for batch in DataLoader(self.val_dataset, 
                                  batch_size=self.config['batch_size']*2):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(inputs['features'])
                loss = self.criterion(outputs, (inputs['emissions'], inputs['total']))
                total_loss += loss.item()
                preds.extend(outputs[1].cpu().numpy())
                trues.extend(inputs['total'].cpu().numpy())
        val_r2 = r2_score(trues, preds)
        return total_loss / len(self.val_dataset), val_r2
    
    def run(self):
        best_r2 = -np.inf
        patience = self.config['patience']
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss, val_r2 = self.validate()
            self.scheduler.step(val_loss)
            
            if val_r2 > best_r2:
                best_r2 = val_r2
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience = self.config['patience']
            else:
                patience -= 1
                if patience <= 0:
                    print("Early stopping activated")
                    break
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val R2: {val_r2:.4f} (Best: {best_r2:.4f})")
            
        joblib.dump({
            'preprocessor': self.train_dataset.preprocessor,
            'feature_names': self.train_dataset.num_cols + self.train_dataset.cat_cols,
            'emission_cols': self.train_dataset.emission_cols
        }, 'training_artifacts.pkl')

# ======================
# 生产环境预测接口
# ======================
class EmissionPredictor:
    def __init__(self, model_path='best_model.pth', artifacts_path='training_artifacts.pkl'):
        artifacts = joblib.load(artifacts_path)
        self.preprocessor = artifacts['preprocessor']
        self.feature_names = artifacts['feature_names']
        self.emission_cols = artifacts.get('emission_cols', [])
        
        self.model = MultiTaskEmissionsModel(
            input_size=len(self.feature_names),
            num_tasks=len(self.emission_cols)
        )
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_input(self, input_data):
        df = pd.DataFrame([{
            col: input_data.get(col, np.nan) 
            for col in self.feature_names
        }])
        for col in df.select_dtypes(include=np.number):
            q1 = np.nanquantile(df[col], 0.25)
            q3 = np.nanquantile(df[col], 0.75)
            df[col] = df[col].clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))
        return self.preprocessor.transform(df)
    
    def predict(self, input_data):
        X = self.preprocess_input(input_data)
        with torch.no_grad():
            tensor_input = torch.FloatTensor(X).to(self.device)
            emissions, total = self.model(tensor_input)
            
        return {
            'total_emission': total.item(),
            'emission_components': dict(zip(
                self.emission_cols,
                emissions.squeeze().tolist()
            )) if self.emission_cols else {}
        }
    
if __name__ == "__main__":
    # 配置参数
    config = {
        'data_path': '工作簿1.xlsx',
        'batch_size': 64,
        'lr': 7e-4,
        'weight_decay': 1e-3,
        'epochs': 200,
        'patience': 25,
        'hidden_dims': [256, 128],
        'possible_cat_cols': ['Battery_Type', 'Region', 'Supplier']
    }
    
    # 训练流程
    trainer = ModelTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.run()
    
    # 示例预测
    predictor = EmissionPredictor()
    sample_data = {
        'Battery_Type': 'LFP',
        'Mixing_anode': 1.5125,
        'Dry_room': 6.798389494,
        'Assembly_Country': 'China'
        # 其他特征缺失时将自动填补
    }
      # 根据实际数据列填充
    prediction = predictor.predict(sample_data)
    print(f"\n预测总碳排量：{prediction['total_emission']:.2f} kgCO₂e")
    if prediction['emission_components']:
        print("分项碳排预测：")
        for k, v in prediction['emission_components'].items():
            print(f"  {k}: {v:.2f}")
