# app.py
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import joblib
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals['zip'] = zip

# 加载LightGBM模型和预处理器
def load_models():
    """加载所有模型及预处理器"""
    try:
        # 加载预处理器和特征名称
        preprocessor = joblib.load('preprocessor.pkl')
        feature_names = joblib.load('feature_names.pkl')
        emission_cols = joblib.load('emission_cols.pkl')
        
        # 加载各分项排放模型
        emission_models = {}
        for col in emission_cols:
            emission_models[col] = joblib.load(f'{col}_model.pkl')
        
        # 加载总排放模型
        total_model_direct = joblib.load('total_model_direct.pkl')
        total_model_ensemble = joblib.load('total_model_ensemble.pkl')
        
        return emission_models, total_model_direct, total_model_ensemble, preprocessor, feature_names, emission_cols
    except Exception as e:
        print(f"模型加载错误: {e}")
        raise e

# 全局加载模型
try:
    emission_models, total_model_direct, total_model_ensemble, preprocessor, feature_names, emission_cols = load_models()
except Exception as e:
    print(f"初始化错误: {e}")
    # 在实际部署中，您可能希望添加一个备用方案或优雅地退出
    # 这里为了简单起见，我们会继续执行，但在请求处理时会检查模型是否成功加载

# 辅助函数：将可选的字符串转换为 float
def to_float(val: Optional[str]) -> Optional[float]:
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的浮点数值: {val}")

# 辅助函数：获取填补后的特征值
def get_imputed_features(df_input, preprocessor, feature_names):
    """获取经过预处理器的imputer填补后的特征值"""
    # 根据训练时的设置，分类特征为：
    cat_cols = ['Battery_Type', 'Component_Country', 'Assembly_Country']
    # 数值特征为除去分类特征的其他所有特征
    num_cols = [feat for feat in feature_names if feat not in cat_cols]
    
    # 创建一个副本用于填补
    df_imputed = df_input.copy()
    
    try:
        # 对分类特征进行缺失值填补
        cat_transformer = preprocessor.named_transformers_['cat']
        cat_imputer = cat_transformer.named_steps['imputer']
        cat_imputed = cat_imputer.transform(df_input[cat_cols])
        
        # 对数值特征进行缺失值填补
        num_transformer = preprocessor.named_transformers_['num']
        num_imputer = num_transformer.named_steps['imputer']
        num_imputed = num_imputer.transform(df_input[num_cols])
        
        # 将填补后的值放回DataFrame
        for i, col in enumerate(cat_cols):
            df_imputed[col] = cat_imputed[0][i]
            
        for i, col in enumerate(num_cols):
            df_imputed[col] = num_imputed[0][i]
            
        # 创建原始值和填补后值的比较表
        comparison_df = pd.DataFrame()
        
        for col in feature_names:
            comparison_df[f"{col}(原始)"] = df_input[col]
            comparison_df[f"{col}(填补后)"] = df_imputed[col]
        
        # 将NaN显示为"未提供"    
        comparison_df = comparison_df.fillna("未提供")
        
        # 标记填补的值
        for col in feature_names:
            mask = (df_input[col].isna()) & (~comparison_df[f"{col}(填补后)"].isin(["未提供"]))
            if mask.any():
                comparison_df.loc[mask, f"{col}(填补后)"] = comparison_df.loc[mask, f"{col}(填补后)"].astype(str) + " (自动填补)"
        
        # 转换成HTML表格
        comparison_html = comparison_df.to_html(classes="table table-striped table-bordered", index=False)
        
        return df_imputed, comparison_html
        
    except Exception as e:
        print(f"特征填补过程中出错: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，返回原始数据和错误信息
        return df_input, f"<div class='alert alert-danger'>特征填补错误: {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def get_prediction(
    request: Request,
    # 字符串字段直接使用 Form
    Battery_Type: str = Form(...),
    Component_Country: str = Form(None),
    Assembly_Country: str = Form(...),
    # 数值字段先接收为 Optional[str]（允许留空）
    Cathode_active_material_1: Optional[str] = Form(""),
    graphite: Optional[str] = Form(""),
    Carbon_black_Cathode: Optional[str] = Form(""),
    Binder_SBR_CMC_Anode: Optional[str] = Form(""),
    Binder_SBR_CMC_Cathode: Optional[str] = Form(""),
    Binder_PVDF_Anode: Optional[str] = Form(""),
    Binder_PVDF_Cathode: Optional[str] = Form(""),
    Copper_Foil: Optional[str] = Form(""),
    Copper_Tab: Optional[str] = Form(""),
    Aluminum_Foil: Optional[str] = Form(""),
    Aluminum_Tab: Optional[str] = Form(""),
    Aluminum_Container: Optional[str] = Form(""),
    Electrolyte_LiPF6: Optional[str] = Form(""),
    Electrolyte_EC: Optional[str] = Form(""),
    Electrolyte_DMC: Optional[str] = Form(""),
    Plastic_PE_Separator: Optional[str] = Form(""),
    Plastic_CPP_Container: Optional[str] = Form(""),
    Plastic_PET_Container: Optional[str] = Form(""),
    Nylon_Carbon_Fiber_Container: Optional[str] = Form(""),
    Mixing_anode: Optional[str] = Form(""),
    Mixing_cathode: Optional[str] = Form(""),
    Coating_anode: Optional[str] = Form(""),
    Coating_cathode: Optional[str] = Form(""),
    Drying_anode: Optional[str] = Form(""),
    Drying_cathode: Optional[str] = Form(""),
    Calendering_anode: Optional[str] = Form(""),
    Calendering_cathode: Optional[str] = Form(""),
    Slitting_anode: Optional[str] = Form(""),
    Slitting_cathode: Optional[str] = Form(""),
    Stacking: Optional[str] = Form(""),
    Filling: Optional[str] = Form(""),
    Formation: Optional[str] = Form(""),
    Floor_heating: Optional[str] = Form(""),
    Dry_room: Optional[str] = Form(""),
    Miscellaneous: Optional[str] = Form(""),
    Drying_anode_heat: Optional[str] = Form(""),
    Drying_cathode_heat: Optional[str] = Form(""),
    Dry_room_heat: Optional[str] = Form(""),
    Brine_Ratio: Optional[str] = Form(""),
    Spodumene_Ratio: Optional[str] = Form("")
):
    try:
        # 将数值字段转换为 float（若为空则返回 None）
        Cathode_active_material_1 = to_float(Cathode_active_material_1)
        graphite = to_float(graphite)
        Carbon_black_Cathode = to_float(Carbon_black_Cathode)
        Binder_SBR_CMC_Anode = to_float(Binder_SBR_CMC_Anode)
        Binder_SBR_CMC_Cathode = to_float(Binder_SBR_CMC_Cathode)
        Binder_PVDF_Anode = to_float(Binder_PVDF_Anode)
        Binder_PVDF_Cathode = to_float(Binder_PVDF_Cathode)
        Copper_Foil = to_float(Copper_Foil)
        Copper_Tab = to_float(Copper_Tab)
        Aluminum_Foil = to_float(Aluminum_Foil)
        Aluminum_Tab = to_float(Aluminum_Tab)
        Aluminum_Container = to_float(Aluminum_Container)
        Electrolyte_LiPF6 = to_float(Electrolyte_LiPF6)
        Electrolyte_EC = to_float(Electrolyte_EC)
        Electrolyte_DMC = to_float(Electrolyte_DMC)
        Plastic_PE_Separator = to_float(Plastic_PE_Separator)
        Plastic_CPP_Container = to_float(Plastic_CPP_Container)
        Plastic_PET_Container = to_float(Plastic_PET_Container)
        Nylon_Carbon_Fiber_Container = to_float(Nylon_Carbon_Fiber_Container)
        Mixing_anode = to_float(Mixing_anode)
        Mixing_cathode = to_float(Mixing_cathode)
        Coating_anode = to_float(Coating_anode)
        Coating_cathode = to_float(Coating_cathode)
        Drying_anode = to_float(Drying_anode)
        Drying_cathode = to_float(Drying_cathode)
        Calendering_anode = to_float(Calendering_anode)
        Calendering_cathode = to_float(Calendering_cathode)
        Slitting_anode = to_float(Slitting_anode)
        Slitting_cathode = to_float(Slitting_cathode)
        Stacking = to_float(Stacking)
        Filling = to_float(Filling)
        Formation = to_float(Formation)
        Floor_heating = to_float(Floor_heating)
        Dry_room = to_float(Dry_room)
        Miscellaneous = to_float(Miscellaneous)
        Drying_anode_heat = to_float(Drying_anode_heat)
        Drying_cathode_heat = to_float(Drying_cathode_heat)
        Dry_room_heat = to_float(Dry_room_heat)
        Brine_Ratio = to_float(Brine_Ratio)
        Spodumene_Ratio = to_float(Spodumene_Ratio)

        # 收集表单数据
        input_data = {
            "Battery_Type": Battery_Type,
            "Component_Country": Component_Country,
            "Assembly_Country": Assembly_Country,
            "Cathode_active_material_1": Cathode_active_material_1,
            "graphite": graphite,
            "Carbon_black_Cathode": Carbon_black_Cathode,
            "Binder_SBR_CMC_Anode": Binder_SBR_CMC_Anode,
            "Binder_SBR_CMC_Cathode": Binder_SBR_CMC_Cathode,
            "Binder_PVDF_Anode": Binder_PVDF_Anode,
            "Binder_PVDF_Cathode": Binder_PVDF_Cathode,
            "Copper_Foil": Copper_Foil,
            "Copper_Tab": Copper_Tab,
            "Aluminum_Foil": Aluminum_Foil,
            "Aluminum_Tab": Aluminum_Tab,
            "Aluminum_Container": Aluminum_Container,
            "Electrolyte_LiPF6": Electrolyte_LiPF6,
            "Electrolyte_EC": Electrolyte_EC,
            "Electrolyte_DMC": Electrolyte_DMC,
            "Plastic_PE_Separator": Plastic_PE_Separator,
            "Plastic_CPP_Container": Plastic_CPP_Container,
            "Plastic_PET_Container": Plastic_PET_Container,
            "Nylon_Carbon_Fiber_Container": Nylon_Carbon_Fiber_Container,
            "Mixing_anode": Mixing_anode,
            "Mixing_cathode": Mixing_cathode,
            "Coating_anode": Coating_anode,
            "Coating_cathode": Coating_cathode,
            "Drying_anode": Drying_anode,
            "Drying_cathode": Drying_cathode,
            "Calendering_anode": Calendering_anode,
            "Calendering_cathode": Calendering_cathode,
            "Slitting_anode": Slitting_anode,
            "Slitting_cathode": Slitting_cathode,
            "Stacking": Stacking,
            "Filling": Filling,
            "Formation": Formation,
            "Floor_heating": Floor_heating,
            "Dry_room": Dry_room,
            "Miscellaneous": Miscellaneous,
            "Drying_anode_heat": Drying_anode_heat,
            "Drying_cathode_heat": Drying_cathode_heat,
            "Dry_room_heat": Dry_room_heat,
            "Brine_Ratio": Brine_Ratio,
            "Spodumene_Ratio": Spodumene_Ratio
        }
        
        # 构造完整输入特征（缺失的以 NaN 表示）
        full_input = {feat: input_data.get(feat, np.nan) for feat in feature_names}
        df_input = pd.DataFrame([full_input])
        
        # 确保DataFrame的列顺序与训练时相同
        df_input = df_input[feature_names]
        
        # ---------------------------
        # 获取填补后的特征值
        # ---------------------------
        df_imputed, comparison_html = get_imputed_features(df_input, preprocessor, feature_names)
        
        # ---------------------------
        # 使用LightGBM模型进行预测
        # ---------------------------
        
        # 预处理输入数据
        X_processed = preprocessor.transform(df_input)
        
        # 预测各个分项排放
        emission_values = {}
        emission_array = []
        for col, model in emission_models.items():
            pred = model.predict(X_processed)[0]
            emission_values[col] = pred
            emission_array.append(pred)
            
        # 组合预测结果作为集成模型的输入
        X_ensemble = np.hstack([X_processed, np.array(emission_array).reshape(1, -1)])
        
        # 使用集成模型预测总量
        total_emission_val = total_model_ensemble.predict(X_ensemble)[0]
        
        # 计算各分项占比
        breakdowns = [(val / total_emission_val * 100) for val in emission_values.values()]
        
        # 排放量和占比表格
        emission_data = pd.DataFrame({
            '排放项': emission_cols,
            '排放量(kgCO₂e)': [emission_values[col] for col in emission_cols],
            '占比(%)': breakdowns
        })
        emission_table_html = emission_data.to_html(
            classes="table table-striped table-bordered", 
            index=False,
            float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x
        )
        
        # 返回模板时将填补后的特征HTML传递过去
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": total_emission_val,
            "breakdowns": breakdowns,
            "emission_names": emission_cols,
            "filled_features": comparison_html,
            "emission_values": [emission_values[col] for col in emission_cols],
            "emission_table": emission_table_html
        })
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)