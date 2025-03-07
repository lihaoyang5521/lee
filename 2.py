import pandas as pd
import numpy as np
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 可重复性设置
# ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    from sklearn import config_context
    with config_context(assume_finite=True, working_memory=128):
        np.random.seed(seed)

set_seed(42)

# ======================
# 数据预处理函数（关键修改点1）
# ======================
def preprocess_data(df, target_col, emission_cols, cat_cols, num_cols):
    """
    预处理数据，包含特征工程和编码
    """
    # 分离特征和目标变量
    X = df.drop(columns=emission_cols + [target_col])
    y_emissions = df[emission_cols]
    y_total = df[target_col]
    
    # 分类特征处理 Pipeline（改用TargetEncoder）
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', TargetEncoder())
    ])
    
    # 数值特征处理 Pipeline
    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('power_transform', PowerTransformer()),
        ('scaler', RobustScaler())
    ])
    
    # 合并处理流程（关键修改点2：禁用特征名前缀）
    preprocessor = ColumnTransformer(
        [
            ('cat', cat_pipeline, cat_cols),
            ('num', num_pipeline, num_cols)
        ],
        verbose_feature_names_out=False
    )
    
    # 拟合和转换数据
    X_processed = preprocessor.fit_transform(X, y_total)
    
    return X_processed, y_emissions, y_total, preprocessor

# ======================
# 模型训练函数（优化参数）
# ======================
def train_gbdt_models(X, y_emissions, y_total, emission_cols):
    """
    训练梯度提升树模型
    """
    # 分项排放模型参数
    emission_models = {}
    for i, col in enumerate(emission_cols):
        print(f"Training model for {col}...")
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42
        )
        
        # 训练验证拆分
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_emissions.iloc[:, i], test_size=0.2, random_state=42
        )
        
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        emission_models[col] = model
    
    # 总量预测模型（直接）
    total_model_direct = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=127,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.2,
        n_jobs=-1,
        random_state=42
    )
    
    X_train_total, X_val_total, y_train_total, y_val_total = train_test_split(
        X, y_total, test_size=0.2, random_state=42
    )
    
    total_model_direct.fit(
        X_train_total, 
        y_train_total,
        eval_set=[(X_val_total, y_val_total)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # 集成模型（关键修改点3：确保特征维度匹配）
    emission_preds = np.column_stack([model.predict(X) for model in emission_models.values()])
    X_ensemble = np.hstack([X, emission_preds])
    
    total_model_ensemble = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42
    )
    
    X_train_ens, X_val_ens, y_train_ens, y_val_ens = train_test_split(
        X_ensemble, y_total, test_size=0.2, random_state=42
    )
    
    total_model_ensemble.fit(
        X_train_ens, 
        y_train_ens,
        eval_set=[(X_val_ens, y_val_ens)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    return emission_models, total_model_direct, total_model_ensemble

# ======================
# 预测函数（关键修改点4）
# ======================
def predict_total_emission(input_data, emission_models, total_model_ensemble, 
                          preprocessor, feature_names, emission_cols):
    """
    预测碳排放量（确保特征完整性）
    """
    # 构造完整输入（包含所有特征）
    full_input = {feat: input_data.get(feat, np.nan) for feat in feature_names}
    
    # 创建数据框并确保列顺序
    df_input = pd.DataFrame([full_input]).reindex(columns=feature_names)
    
    # 预处理输入数据
    X_processed = preprocessor.transform(df_input)
    
    # 预测分项排放
    emission_values = {}
    emission_array = []
    for col, model in emission_models.items():
        pred = model.predict(X_processed)[0]
        emission_values[col] = pred
        emission_array.append(pred)
    
    # 组合特征
    X_ensemble = np.hstack([X_processed, np.array(emission_array).reshape(1, -1)])
    
    # 预测总量
    total_emission = total_model_ensemble.predict(X_ensemble)[0]
    
    return {
        "total_emission": total_emission,
        "emission_breakdown": emission_values
    }

# ======================
# 主流程（添加特征验证）
# ======================
def main():
    # 数据加载
    data_path = '工作簿2.xlsx'
    df = pd.read_excel(data_path)
    
    # 定义列
    target_col = 'Total_Carbon_Emissions'
    emission_cols = [c for c in df.columns if c.endswith('_Carbon_Emissions') and c != target_col]
    cat_cols = ['Battery_Type', 'Component_Country', 'Assembly_Country']
    num_cols = [c for c in df.columns if c not in cat_cols + emission_cols + [target_col]]
    
    # 打印特征信息
    print("原始特征数量:", len(cat_cols) + len(num_cols))
    
    # 数据拆分
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 预处理
    X_train, y_emissions_train, y_total_train, preprocessor = preprocess_data(
        train_df, target_col, emission_cols, cat_cols, num_cols
    )
    
    # 获取特征名称（关键修改点5）
    feature_names = preprocessor.get_feature_names_out()
    print("\n预处理后的特征名称:", feature_names)
    
    # 训练模型
    emission_models, total_model_direct, total_model_ensemble = train_gbdt_models(
        X_train, y_emissions_train, y_total_train, emission_cols
    )
    
    # 保存模型（包含特征名称）
    joblib.dump({
        'emission_models': emission_models,
        'total_model_ensemble': total_model_ensemble,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'emission_cols': emission_cols
    }, 'carbon_emission_models.pkl')
    
    # 示例预测（关键修改点6：完整特征输入）
    sample_input = {
        'Battery_Type': 'LFP',
        'Mixing_anode': 1.5125,
        'Dry_room': 6.798389494,
        'Assembly_Country': 'China',
        # 其他特征需要显式声明为NaN
        'Brine_Ratio': np.nan,
        'Aluminum_Foil': np.nan,
        # ...（根据实际特征列表补充所有特征）
    }
    
    # 自动补全缺失特征
    complete_input = {feat: sample_input.get(feat, np.nan) for feat in feature_names}
    
    prediction = predict_total_emission(
        complete_input, 
        emission_models, 
        total_model_ensemble,
        preprocessor,
        feature_names,
        emission_cols
    )
    
    print("\n预测结果:")
    print(f"总碳排放: {prediction['total_emission']:.2f} kgCO₂e")
    for k, v in prediction['emission_breakdown'].items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()