from sklearn.linear_model import LinearRegression
from .core import VZVHungaryBaselineModel

class LinearRegressionModel(VZVHungaryBaselineModel):
    """Linear Regression baseline model"""
    
    def __init__(self, 
                 dataloader,
                 fit_intercept=True,
                 updates=False):
        
        model_params = {
            'fit_intercept': fit_intercept
        }
        
        super().__init__(
            model_class=LinearRegression,
            dataloader=dataloader,
            model_params=model_params,
            updates=updates
        )
