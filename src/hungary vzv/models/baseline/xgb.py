import xgboost as xgb
from .core import VZVHungaryBaselineModel

class XGBoostModel(VZVHungaryBaselineModel):
    """XGBoost baseline model"""
    
    def __init__(self, 
                 dataloader,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=42,
                 n_jobs=-1,
                 updates=False):
        
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'reg:squarederror'
        }
        
        super().__init__(
            model_class=xgb.XGBRegressor,
            dataloader=dataloader,
            model_params=model_params,
            updates=updates
        )
