from sklearn.ensemble import RandomForestRegressor
from .core import VZVHungaryBaselineModel

class RandomForestModel(VZVHungaryBaselineModel):
    """Random Forest baseline model"""
    
    def __init__(self, 
                 dataloader,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=42,
                 n_jobs=-1,
                 updates=False):
        
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        super().__init__(
            model_class=RandomForestRegressor,
            dataloader=dataloader,
            model_params=model_params,
            updates=updates
        )
