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

    
    std_data = np.zeros_like(data)
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