import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd

def standardize_casenumbers(data: np.ndarray, method: str = 'zscore', params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Standardize data by county using various normalization methods.
    
    Parameters:
    ----------
    data : np.ndarray
        input data to be standardized. Shape: [n_counties, n_timepoints]
    method : str
        normalization method to use. Options include: ['zscore', 'minmax', 'log']
    params : Optional[Dict]
        Optional dictionary with pre-computed parameters for standardization.
        If not provided, parameters will be computed.
        Used to standardize validation and testing data using the same parameters as training data.

    Returns:
    -------
    standardized_data : np.ndarray
        Standardized data. Shape: [n_counties, n_timepoints]
    params : Dict
        Dictionary with pre-computed parameters for standardization.
    """
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        n_counties = data.shape[0]
        
        df = pd.DataFrame()
        for i in range(n_counties):
            county_df = pd.DataFrame({
                'county_tk': i,
                'cases': data[i, :]
            })
            df = pd.concat([df, county_df], ignore_index=True)

    else:
        raise ValueError('please supply data in a np.ndarray format with [n_counties, n_timepoints]')
    
    standardized_df, standardization_params = standardize_timeseries(
        df, method=method, by_county=True, params=params
    )

    
    std_data = np.zeros_like(data, dtype = np.float32)
    for i in range(n_counties):
        county_data = standardized_df[standardized_df['county_tk'] == i]['cases'].values
        std_data[i, :len(county_data)] = county_data
    
    return std_data, standardization_params

def standardize_timeseries(
    df: np.ndarray,
    method: str = 'zscore',
    by_county: bool = True,
    params: Optional[Dict[Any, Dict[Any, Any]]] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Dict[Any, Any]]:
    """
    Standardize time series data using specified method.

    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Time series data to standardize. If DataFrame, should have 'county_tk' and 'cases' columns.
        If ndarray, shape should be (n_counties, n_timepoints) if by_county=True.
    method : str, default='zscore'
        Standardization method: 'zscore', 'minmax', or 'log'.
    by_county : bool, default=True
        Whether to standardize per county or globally.
    params : dict, optional
        Predefined parameters to apply existing standardization scales.

    Returns:
    --------
    tuple
        (standardized_data, params)
    """

    if params is None:
        params = {}
    # Initialize params dictionary if None
    # Standardize data
    if by_county:
        counties            = df['county_tk'].unique()
        std_data            = df.copy()
        std_data['cases']   = std_data['cases'].astype(float)

        for county in counties:
            county_mask = df['county_tk'] == county
            county_data = df.loc[county_mask, 'cases']
            

            if params and county in params:
                county_params = params[county]
                county_method = county_params.get('method', method)
            
            else: 
                county_method = method

            if county_method == 'zscore':
                center                              = county_params.get('center', 0) if county in params else county_data.mean()
                scale                               = county_params.get('scale', 1)  if county in params else county_data.std()
                std_data.loc[county_mask, 'cases']  = (county_data - center) / (scale + 1e-8)
                if county not in params:
                    params[county] = {'center': center, 'scale': scale, 'method': 'zscore'}
                    

            elif county_method == 'minmax':
                min_val                             = county_params.get('min', 0) if county in params else county_data.min()
                max_val                             = county_params.get('max', 1) if county in params else county_data.max()
                std_data.loc[county_mask, 'cases']  = (county_data - min_val) / (max_val - min_val + 1e-8)
                if county not in params:
                    params[county] = {'min': min_val, 'max': max_val, 'method': 'minmax'}

            elif county_method == 'log':
                std_data.loc[county_mask, 'cases'] = np.log1p(county_data)
                if county not in params:
                    params[county] = {'method': 'log'}

            else:
                raise ValueError(f"Unknown method '{county_method}' in params for county {county}")
    
    else:
        raise ValueError("Global standardization is not supported.")


    return std_data, params

def reconstruct_timeseries(
    standardized_data: np.ndarray,
    params,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Reconstruct the original time series data from standardized data.

    Parameters:
    -----------
    standardized_data : np.ndarray
        Standardized data. Shape: [n_counties, n_timepoints]
    params : Dict
        Dictionary containing standardization parameters for each county.
    method : str
        Standardization method: 'zscore', 'minmax', or 'log'.

    Returns:
    --------
    np.ndarray
        Reconstructed original time series data. Shape: [n_counties, n_timepoints]
    """
    n_counties, n_timepoints = standardized_data.shape
    reconstructed_data = np.zeros_like(standardized_data)

    for i in range(n_counties):
        county_data = standardized_data[i, :]
        county_params = params.get(i, {})
        county_method = county_params.get('method', method)

        if county_method == 'zscore':
            center = county_params.get('center', 0)
            scale = county_params.get('scale', 1)
            reconstructed_data[i, :] = county_data * scale + center

        elif county_method == 'minmax':
            min_val = county_params.get('min', 0)
            max_val = county_params.get('max', 1)
            reconstructed_data[i, :] = county_data * (max_val - min_val) + min_val

        elif county_method == 'log':
            reconstructed_data[i, :] = np.expm1(county_data)

        else:
            raise ValueError(f"Unknown method '{county_method}' for county {i}")

    return reconstructed_data

import numpy as np
import pandas as pd
from typing import Tuple, Optional

def create_cyclical_features(values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert cyclical values to sine/cosine features
    
    Parameters:
    -----------
    values : np.ndarray
        The cyclical values (e.g., week numbers, day of year)
    period : int
        The period of the cycle (e.g., 52 for weeks, 365 for days)
    
    Returns:
    --------
    sin_features : np.ndarray
        Sine transformed features
    cos_features : np.ndarray  
        Cosine transformed features
    """
    # Normalize to [0, 2π]
    normalized = 2 * np.pi * values / period
    
    sin_features = np.sin(normalized)
    cos_features = np.cos(normalized)
    
    return sin_features, cos_features

def add_temporal_features(data_timepoints: pd.DataFrame, 
                         include_week_of_year: bool = True,
                         include_week_absolute: bool = True,
                         include_month: bool = True,
                         include_quarter: bool = True) -> pd.DataFrame:
    """
    Add various temporal features with cyclical encoding
    
    Parameters:
    -----------
    data_timepoints : pd.DataFrame
        DataFrame with temporal information (should have 'week_abs', 'year', 'week_rel' columns)
    include_week_of_year : bool
        Include week of year (1-52) as sin/cos features
    include_week_absolute : bool
        Include absolute week number as linear feature
    include_month : bool
        Include month as sin/cos features
    include_quarter : bool
        Include quarter as sin/cos features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = data_timepoints.copy()
    
    if include_week_of_year:
        # Week of year (1-52 cycle)
        week_of_year = df['week_rel'] + 1  # Convert 0-based to 1-based
        week_sin, week_cos = create_cyclical_features(week_of_year, 52)
        df['week_sin'] = week_sin
        df['week_cos'] = week_cos
    
    if include_week_absolute:
        # Absolute week (linear trend)
        df['week_abs_norm'] = (df['week_abs'] - df['week_abs'].min()) / (df['week_abs'].max() - df['week_abs'].min())
    
    if include_month:
        # Approximate month from week_rel (assuming 4.33 weeks per month)
        month_approx = (df['week_rel'] / 4.33).astype(int) + 1
        month_approx = np.clip(month_approx, 1, 12)  # Ensure valid month range
        month_sin, month_cos = create_cyclical_features(month_approx, 12)
        df['month_sin'] = month_sin
        df['month_cos'] = month_cos
    
    if include_quarter:
        # Quarter from week_rel (13 weeks per quarter)
        quarter = (df['week_rel'] // 13) + 1
        quarter = np.clip(quarter, 1, 4)
        quarter_sin, quarter_cos = create_cyclical_features(quarter, 4)
        df['quarter_sin'] = quarter_sin
        df['quarter_cos'] = quarter_cos
    
    return df

def create_temporal_feature_matrix(temporal_df: pd.DataFrame, 
                                 feature_columns: list = None) -> np.ndarray:
    """
    Create a matrix of temporal features for each time point
    
    Parameters:
    -----------
    temporal_df : pd.DataFrame
        DataFrame with temporal features
    feature_columns : list
        List of column names to include as features
        
    Returns:
    --------
    np.ndarray
        Matrix of shape (n_timepoints, n_features)
    """
    if feature_columns is None:
        # Default temporal features
        feature_columns = [col for col in temporal_df.columns 
                          if any(suffix in col for suffix in ['_sin', '_cos', '_norm'])]
    
    return temporal_df[feature_columns].values
