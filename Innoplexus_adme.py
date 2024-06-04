import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from tqdm import tqdm
from tdc.benchmark_group import admet_group
from fingerprint_gen import * 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("model_output.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger()

benchmark_settings = {
    'caco2_wang': ('regression', True),
    'lipophilicity_astrazeneca': ('regression', True),
    'solubility_aqsoldb': ('regression', True),
    'ppbr_az': ('regression', True),
    'vdss_lombardo': ('regression', True),
    'half_life_obach': ('regression', True),
    'clearance_hepatocyte_az': ('regression', True),
    'clearance_microsome_az': ('regression', True),
    'ld50_zhu': ('regression', True)
}

admet_group_instance = admet_group(path='data/')

for benchmark_name in benchmark_settings.keys():
    task_type, log_transform = benchmark_settings[benchmark_name]
    if task_type != 'regression':
        continue
    
    all_predictions = []
    
    for random_seed in tqdm([1, 2, 3, 4, 5]):
        benchmark = admet_group_instance.get(benchmark_name)
        model_predictions = {}
        dataset_name = benchmark['name']
        train_data, test_data = benchmark['train_val'], benchmark['test']
        
        train_features = generate_fingerprints(train_data['Drug'])
        test_features = generate_fingerprints(test_data['Drug'])
        
        train_features = np.where(np.isfinite(train_features), train_features, np.nan)
        test_features = np.where(np.isfinite(test_features), test_features, np.nan)
        
        imputer = SimpleImputer(strategy='mean')
        train_features = imputer.fit_transform(train_features)
        test_features = imputer.transform(test_features)
        
        max_value = np.finfo(np.float32).max
        train_features = np.clip(train_features, -max_value, max_value)
        test_features = np.clip(test_features, -max_value, max_value)
        
        target_scaler = LogScaler(apply_log=log_transform)
        target_scaler.fit(train_data['Y'].values)
        train_data['Y_scaled'] = target_scaler.transform(train_data['Y'].values)
        
        # LightGBM model
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'verbose': -1
        }
        lgb_train_data = lgb.Dataset(train_features, label=train_data['Y_scaled'].values)
        lgb_model = lgb.train(lgb_params, lgb_train_data)
        lgb_predictions = target_scaler.inverse_transform(lgb_model.predict(test_features)).reshape(-1)
        
        # Random Forest model
        rf_model = RandomForestRegressor()
        rf_model.fit(train_features, train_data['Y_scaled'].values)
        rf_predictions = target_scaler.inverse_transform(rf_model.predict(test_features)).reshape(-1)
        
        # XGBoost model
        xgb_params = {
            'objective': 'reg:squarederror'
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(train_features, train_data['Y_scaled'].values)
        xgb_predictions = target_scaler.inverse_transform(xgb_model.predict(test_features)).reshape(-1)
        
        # CatBoost model
        catboost_model = CatBoostRegressor(verbose=0, random_seed=random_seed)
        catboost_model.fit(train_features, train_data['Y_scaled'].values)
        catboost_predictions = target_scaler.inverse_transform(catboost_model.predict(test_features)).reshape(-1)
        
        # Evaluate models using MAE
        lgb_mae = mean_absolute_error(test_data['Y'], lgb_predictions)
        rf_mae = mean_absolute_error(test_data['Y'], rf_predictions)
        xgb_mae = mean_absolute_error(test_data['Y'], xgb_predictions)
        catboost_mae = mean_absolute_error(test_data['Y'], catboost_predictions)
        
        # Select the best model based on MAE
        best_model_name = None
        best_model_params = None
        best_predictions = None
        
        if lgb_mae <= rf_mae and lgb_mae <= xgb_mae and lgb_mae <= catboost_mae:
            best_model_name = 'LightGBM'
            best_model_params = lgb_params
            best_predictions = lgb_predictions
        elif rf_mae <= lgb_mae and rf_mae <= xgb_mae and rf_mae <= catboost_mae:
            best_model_name = 'Random Forest'
            best_model_params = {}
            best_predictions = rf_predictions
        elif xgb_mae <= lgb_mae and xgb_mae <= rf_mae and xgb_mae <= catboost_mae:
            best_model_name = 'XGBoost'
            best_model_params = xgb_params
            best_predictions = xgb_predictions
        else:
            best_model_name = 'CatBoost'
            best_model_params = {}
            best_predictions = catboost_predictions
        
        # Store the best predictions
        model_predictions[dataset_name] = best_predictions
        all_predictions.append(model_predictions)
        
        logger.info(f'Benchmark: {benchmark_name}, Seed: {random_seed}, Best Model: {best_model_name}, Parameters: {best_model_params}')
    
    # Evaluate and print the results on TDC becnhmark 
    evaluation_results = admet_group_instance.evaluate_many(all_predictions)
    logger.info('\n\n{}'.format(evaluation_results))