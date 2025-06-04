from utils import DataProcessingOrchestrator, dir_casedata_ge, bundeslanden_token
import pandas as pd
import os

counter = 0
for ind, f in enumerate(os.listdir(dir_casedata_ge)):
    if f.endswith('.csv'):
        counter = counter + 1
        singledata  = DataProcessingOrchestrator(name = 'single file')
        singledata.import_data(filename = f, directory = dir_casedata_ge, colnames_row=1, separator = "\t", encoding = 'utf-16')
        singledata.drop(labels = slice(0,1), axis = 0)
        singledata.rename_cols({'Unnamed: 0':'week'})
        singledata.mutate('year',value = int(f[-8:-4]))
        if counter == 1:
            fulldata = singledata.df
        else:
            fulldata = pd.concat([fulldata, singledata.df])

fulldata_og = DataProcessingOrchestrator(fulldata)
fulldata = (
    fulldata_og
    .drop(labels = 'Total', axis = 1)
    .pivot_longer(index = ['week','year'],levels_from =  list(bundeslanden_token.keys()), value_colname = 'cases', levels_colname = 'bundesland')
    .filter(conditions = [('bundesland','Unknown','!=')])
    .impute(colnames = 'cases', method = 'zero')
    .replace(colname = 'bundesland', mapping_dict = bundeslanden_token)
    .rename_cols({'bundesland':'county_tk','week':'week_rel'})
    .change_dtype(dtype_dict = {'week_rel': 'int64'})
    .filter(conditions = [('year',2020,'<')])
    
)       
fulldata.df['week_rel'] = fulldata.df['week_rel']-1
fulldata.df['week_abs'] = fulldata.df.groupby('county_tk').cumcount()

main_df         = fulldata.copy().pivot_wider(index = ['week_abs','year','week_rel'], cols_from = 'county_tk', values_from = 'cases')
data_timepoints = main_df.df[['week_abs','year','week_rel']]
data_to_load    = main_df.df.drop(columns = ['week_abs','year','week_rel']).T.to_numpy()