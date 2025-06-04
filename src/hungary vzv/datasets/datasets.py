from utils import *

######################
# casedata > pytorch #
######################
pytorch_dataloader          = ChickenpoxDatasetLoader()
pytorch_dataset             = pytorch_dataloader.get_dataset()
pytorch_dataset_pd          = pd.DataFrame(pytorch_dataset.targets)
pytorch_dataset_pd.columns  = hu_counties
pytorch_edge_index, pytorch_edge_weight = torch.tensor(pytorch_dataset.edge_index, dtype= torch.int64), torch.tensor(pytorch_dataset.edge_weight, dtype = torch.float32)

#####################
# casedata > github #
#####################
hu_cases_long = (
    DataProcessingOrchestrator(name = 'raw vzv cases')
    .import_data(filename = 'vzv_cases.csv', directory = dir_casedata_hu)
    .change_dtype({'Date': 'datetime64[ns]'})
    .pivot_longer(index = 'Date', levels_from = list(hu_mapping_dict_countynames.keys()), value_colname = 'cases', levels_colname = 'county_name')
    .replace(colname = 'county_name', mapping_dict = hu_mapping_dict_countynames)    
)

fulldata = (
    hu_cases_long.copy()
    .mutate(new_colname = 'county_tk', value = hu_cases_long.df['county_name'])
    .replace(colname = 'county_tk', mapping_dict= hu_countynames_tokens)    
)

fulldata.df['year']        = fulldata.df.Date.dt.year
fulldata.df['week_abs']    = fulldata.df.groupby('county_name').cumcount()
fulldata.df['week_rel']    = fulldata.df.groupby(['county_name','year']).cumcount()

main_df         = fulldata.copy().pivot_wider(index = ['week_abs','year','week_rel'], cols_from = 'county_tk', values_from = 'cases')
data_timepoints = main_df.df[['week_abs','year','week_rel']]
data_to_load    = main_df.df.drop(columns = ['week_abs','year','week_rel']).T.to_numpy()
