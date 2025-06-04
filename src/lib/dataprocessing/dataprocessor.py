import pandas as pd
import geopandas as gpd
from typing import Optional, Dict, Union, List, Tuple
import os
from .filtering import apply_condition
from shapely.geometry import Point, LineString

class DataProcessingOrchestrator:
    """ 
    This class represents the main class used for the processing (and preprocessing) of dataframes.
    This orchestrator delegates work to the specific classes defined below, loaded in `_setup_modules`.
    You can use this class chaining different methods. These will be saved in the `method_registry` attribute.
    This log of methods and parameters can be saved into a YAML-file, using `save_processing_steps`.
    Such a YAML-file may be executed using the config manager, specifically using the method `execute_from_yaml`.

    Parameters
    ----------
    df: 
        An optional pd.DataFrame. If unspecified, the `read_data` method should be among the first ones called
    name: 
        The name internally used while processing. This has no further consequences on the storing of the actual data.
    category: 
        The category into which the data belongs. This defines the subdirectory into which the YAML-file is stored. If unspecified, the log is stored in the main config directory.

    Representation
    -------------
    - the name of the dataframe being processed
    - the shape of the dataframe being processed
    - a preview of the dataframe being processed

    """
    def __init__(self, df: Union[pd.DataFrame, gpd.GeoDataFrame] = None, category: str = None ,name: Optional[str] = "unnamed"):
        self.df             = df
        self.name           = name
        self.realm          = 'dataprocessing'
        self.category       = category
        self.saved_path     = None
        self.status         = 0
        self.method_registry= []

        if self.df is not None:                         # if df is not defined, no use in loading subclasses
            self._setup_modules()
            self.status = 1

    def __repr__(self):
        if self.df is not None:
            return f'DataFrame: {self.name}\nShape: {self.df.shape}\nPreview\n{self.df.head()}'
        else:
            return self.status_update()
        
    def _setup_modules(self):
        """Initializes all data processing modules."""
        pass

    def register_step(self, method: str, variables: Dict):
        """
        Updates method_registry for self. This list keeps track of all methods called with which parameters. 
        Each relevant method updates the registry intrinsically.
        
        Parameters
        ----------
        method: str
            Name of the method used
        variables: Dict
            The dictionary with parameters associated with the method just called
        """
        newstep = {method: variables}
        self.method_registry.append(newstep)
        if self.status > 2:
            self.status = 2.5
        else:
            self.status = 2

    def status_update(self):
        """
        Prints a status update of the DataFrame processing state.
        
        Parameters
        ----------
        None

        Returns
        -------
        DataProcessingOrchestrator
            Returns self for method chaining         

        Examples
        --------
        >>> popsize.status_update()

        Notes
        -----
        Step-registration: no
        """ 

        status_options = {
                0: f'⚠️ no dataframe found',
                1: f'🔄 dataframe {self.name} loaded',
                2: f'🛠️ dataframe {self.name} under process, {len(self.method_registry)} processing steps undertaken',
                2.5: f'⚠️ dataframe {self.name} has been modified since saving!',
                3: f'✅ dataframe {self.name} saved'
            }
        print(status_options[self.status])
        return self
    
    def import_data(self, filename: str, directory: str, dtype_dict: Optional[Dict] = None, separator: str = ",", colnames_row: int = 0, encoding: str = 'utf_8'):
        """
        Reads a datafile from joining directory and filename.
        If self.df is already initialized (i.e. not None) then the dataframes get merged.
        If an .xslx is fed in, then a .csv is saved in one directory down under the same name

        Parameters
        ----------
        filename: str
            The name of the file to be read. Include the extension, since anything ending on '.shp' will be read into a gpd.GeoDataFrame.
        directory: str
            The directory in which the file is saved.
        dtype_dict: dict 
            Dictionary contaninig the column names in keys and datatypes in values.
        separator: str, optional
            The separator used in the datafile. By default ',' but for standard German .csv-files this will be semicolon
        colnames_row: int, optional
            Row number in which the column-names are listed. By default 0.
        encoding: 
            The encoding in which the file is encoded. The default is 'utf-8'. For German text, use 'ISO-8859-1'

        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame

        Examples
        --------
        >>> popsize = (
        >>>     DataProcessingOrchestrator(name = 'population_size')
        >>>     import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>     )


        Notes
        -----
        Step-registration: yes
        """
        vars            = locals()
        del vars['self']
        funcname        = 'import_data'
        self.register_step(funcname, vars)  

        filepath    = os.path.join(directory, filename)
        extension   = filename.split(".", 1)[1]

        if extension == 'shp':
            newdf = gpd.read_file(filepath, dtype=dtype_dict, encoding=encoding)
        elif extension == 'xlsx':
            newdf       = pd.read_excel(filepath, dtype = dtype_dict, engine = 'openpyxl')
            newfilename = filename.split(".", 1)[0]+".csv"
            newfilepath = os.path.join(os.path.dirname(directory), newfilename)
            newdf.to_csv(newfilepath, sep = ',', index = False)
            print(f'{self.name} loaded from .xlsx has been saved as csv: {newfilepath}')
        else:
            newdf = pd.read_csv(filepath, dtype=dtype_dict, sep=separator, header=colnames_row, encoding=encoding, na_values=['NaN', 'NULL', 'N/A'])

        if self.df is None:
            self.df = newdf
            self._setup_modules()

        else:
            cols1, cols2 = list(self.df.columns), list(newdf.columns)
            if cols1 != cols2:
                raise ValueError(f'merging self.df with newdf failed. self.df with columns {cols1} does not have the same columns as newdf with {cols2}')
            df_merged = pd.concat([self.df, newdf], ignore_index=True)
            self.df = df_merged
        self.status = 1
        return self

    def save_data(self, filename: str, directory: str):
        """
        Saves a datafile from joining directory and filename

        Parameters
        ----------
        filename: str
            The name of the file to be saved. Include the extension, since the saving-methodology is inferred. Extensions are currently limited to: ['.csv', '.tsv', '.shp']
        directory: str
            The directory in which the file is saved.

        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame

        Examples
        --------
        >>> popsize = (
        >>>     DataProcessingOrchestrator(name = 'population_size')
        >>>    .import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>    .pivot_longer(index = ['country', 'country_code'], 
        >>>                levels_from = [str(x) for x in range(1960,2024)],
        >>>                value_colname = 'population_size',
        >>>                levels_colname= 'year')
        >>>    .impute(colnames = 'population_size', method = 'drop')
        >>>    .change_dtype({'year': 'int',
        >>>                'population_size': 'int'})
        >>>    .replace(colname = 'country', mapping_dict= map_popsize_data_countrynames)
        >>>    .filter(conditions = [('country',countries_in_the_world,'in')])
        >>>    .filter(conditions = [('year',datayears[0],">="), ('year',datayears[1],'<=')])
        >>>    .save_data(filename = 'population_size.csv', directory = os.path.join(dir_data_processed, 'sociodemography'))
        >>> )

        Notes
        -----
        Step-registration: yes
        """        
        vars            = locals()
        del vars['self']
        funcname        = 'save_data'
        self.register_step(funcname, vars)          
        
        extension = filename.split(".", 1)[1]
        extensions_separators = {'csv': ',',
                                 'tsv': '\t',
                                 'shp': ''
                             }

        if extension not in list(extensions_separators.keys()):
            raise ValueError(f'{extension} not supported. Please provide one of the following extensions: {list(extensions_separators.keys())}')
        
        path = os.path.join(directory, filename)

        if extension == 'shp':
            if not isinstance(self.df, gpd.GeoDataFrame):
                self.df = gpd.GeoDataFrame(self.df, geometry = 'geometry')
            self.df.to_file(path, index=False)

        else:
            self.df.to_csv(path, sep = extensions_separators[extension], index = False)

        self.status = 3
        return self
    
    def copy(self, as_instance: bool = True):
        """
        Copies the DataFrame and returns it as a new DataProcessingOrchestrator or as a plain DataFrame.

        Parameters
        ----------
        as_instance: str
            Whether to return a DataProcessingOrchestrator instance or a plain DataFrame.

        Returns
        -------
        DataProcessingOrchestrator (as_instance = True) or pd.DataFrame (as_instace = False)

        Examples
        --------
        >>> processor.copy(as_instance = True, newname = 'new processor')
        >>> processor.copy(as_instance = False)

        Notes
        -----
        Step-registration: no
        """
        if as_instance:
            if isinstance(self, DataProcessingOrchestrator):
                new_name = self.name + '_copy'
                return DataProcessingOrchestrator(df=self.df.copy(), name=new_name)
            else:
                raise ValueError(f'Attempted to copy {self.name} as an instance of DataPRocessingOrchestrator, which it is not!')
        else:
            return self.df.copy()
        
    def pivot_wider(self, index: Union[List[str], str], cols_from: str , values_from: str, reset_index: bool = True):
        """
        Pivots df (pd.pivot): separates a column into multiple columns. This is the antithesis of self.pivot_longer.

        Parameters
        ----------
        index: Union[str, List[str]]
            The column or list of columns to be used as indices
        columns_from: str
            The name of the column from which the different columns will be constructed
        values_from: str
            The name of the column from which these newly created columns will be filled
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame

        Examples
        --------
        >>> monthly_ev_cases.pivot_wider(index = 'time', columns_from = 'df_set', values_from = 'cases')

        Notes
        -----
        Step-registration: yes
        """        
        vars            = locals()
        del vars['self']
        funcname        = 'pivot_wider'
        self.register_step(funcname, vars)    
        if reset_index:
            self.df = self.df.pivot(index = index, columns = cols_from, values = values_from).reset_index()
        else:
            self.df = self.df.pivot(index = index, columns = cols_from, values = values_from)
        self.register_step(funcname, vars) 
        return self

    def pivot_longer(self, index: Union[List[str], str], levels_from: Union[List[str], str], value_colname: str, levels_colname: str):
        """
        Melts df (pd.melt); merges columns into one. This is the antithesis of self.pivot_wider.

        Parameters
        ----------
        index: Union[str, List[str]]
            The column or list of columns to be used as indices
        levels_from: Union[str, List[str]]
            Column or list of columns to unpivot
        value_column_name: str
            The name of new column, of which the value will come from the unpivotted columns
        levels_column_name: str
            The name of the new column, of which the levels_from columns will be the new levels
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame

        Examples
        --------
        >>> popsize = (
        >>>    DataProcessingOrchestrator(name = 'population_size')
        >>>    .import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>    .pivot_longer(index = ['country', 'country_code'], 
        >>>                levels_from = [str(x) for x in range(1960,2024)],
        >>>                value_colname = 'population_size',
        >>>                levels_colname= 'year')
        >>>     )


        Notes
        -----
        Step-registration: yes
        """   
        vars            = locals()
        del vars['self']
        funcname        = 'pivot_longer'
        self.register_step(funcname, vars) 

        self.df = pd.melt(self.df, id_vars = index, value_vars = levels_from, value_name=value_colname, var_name=levels_colname)
        return self

    def change_dtype(self, dtype_dict: Dict[str,str]):
        """
       Changes datatype of specified columns

        Parameters
        ----------
        dtype_dict: Dict[str,str]
            A dictionary with column names (keys) and the new datatypes (values)
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame

        Examples
        --------
        >>> popsize = (
        >>>     DataProcessingOrchestrator(name = 'population_size')
        >>>     .import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>     .pivot_longer(index = ['country', 'country_code'], 
        >>>            levels_from = [str(x) for x in range(1960,2024)],
        >>>            value_colname = 'population_size',
        >>>            levels_colname= 'year')
        >>>     .impute(colnames = 'population_size', method = 'drop')
        >>>     .change_dtype({'year': 'int',
        >>>              'population_size': 'int'})
        >>> )


        Notes
        -----
        Step-registration: yes
        """      
        vars            = locals()
        del vars['self']
        funcname        = 'change_dtype'
        self.register_step(funcname, vars)

        for colname, dtype in dtype_dict.items():
            if dtype in ['int','Int64', 'float']:
                self.df[colname] = pd.to_numeric(self.df[colname], errors='coerce')
            self.df[colname] = self.df[colname].astype(dtype)

        self.df = self.df.astype(dtype_dict)
        return self
    
    def impute(self, colnames: Union[List, str], method: str):
        """
        Imputes; deals with NaN values.
        Currently, two methods are supported; 'drop' and 'zero'.

        Parameters
        ----------
        colnames: Union[List, str]
            The column or list of columns to be imputed
        method: str
            The method with which to impute. Currently limited to: ['drop', 'zero']
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> popsize = (
        >>>        DataProcessingOrchestrator(name = 'population_size')
        >>>        .import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>        .pivot_longer(index = ['country', 'country_code'], 
        >>>                    levels_from = [str(x) for x in range(1960,2024)],
        >>>                    value_colname = 'population_size',
        >>>                    levels_colname= 'year')
        >>>        .impute(colnames = 'population_size', method = 'zero')
        >>>     )

        Notes
        -----
        Step-registration: yes
        """

        vars            = locals()
        del vars['self']
        funcname        = 'impute'
        self.register_step(funcname, vars)   

        impute_options = ['drop','zero']
        colnames        = self._check_for_col(colnames)
        if method not in impute_options:
             raise ValueError(f'{method} not a valid imputing option. Valid options are: {impute_options}')
        
        if method == 'drop':
            self.df = self.df.dropna(subset = colnames)

        if method == 'zero':
            self.df[colnames] = self.df[colnames].fillna(0)
  
        return self
    
    def replace(self, colname: str, mapping_dict: Dict[any,any]):
        """
        Replaces values in a single column using a dictionary of reference values.

        Parameters
        ----------
        colname: str
            The column to be replaced
        mapping_dict: Dict[any,any]
           The dictionary with which to replace the column, old values (keys) and new values (values).
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> popsize = (
        >>>        DataProcessingOrchestrator(name = 'population_size')
        >>>        .import_data(filename = 'population_per_country.csv', directory = os.path.join(dir_data_raw, 'sociodemography'))
        >>>        .pivot_longer(index = ['country', 'country_code'], 
        >>>                    levels_from = [str(x) for x in range(1960,2024)],
        >>>                    value_colname = 'population_size',
        >>>                    levels_colname= 'year')
        >>>        .change_dtype({'year': 'int', 'population_size': 'int'})
        >>>        .impute(colnames = 'population_size', method = 'zero')
        >>>        .replace(colname = 'country', mapping_dict= map_popsize_data_countrynames)
        >>>     )

        Notes
        -----
        Step-registration: yes
        """
        vars            = locals()
        del vars['self']
        funcname        = 'replace'
        self.register_step(funcname, vars) 

        self.df[colname] = self.df[colname].replace(mapping_dict)
        return self

    def drop(self, labels: Union[str, List[str], int, List[int]], axis: int):
        """
        Drops specified columns or rows

        Parameters
        ----------
        labels: Union[str, List[str], int, List[int]]
            The rows (Union[int, List[int]], axis = 0) or columns (Union[str, List[str]], axis = 1) to be dropped.
        axis: int
            The axis on which to drop; 0 for dropping rows, 1 for dropping columns.
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> popsize.drop('year',1)
        >>> iata_airtravel.drop(labels = slice(-11,None), axis = 0)

        Notes
        -----
        Step-registration: yes        
        """

        
        vars            = locals()
        del vars['self']
        funcname        = 'drop'
        self.register_step(funcname, vars) 

        if axis == 0:
            if not isinstance(labels, slice):
                raise ValueError(f'when dropping rows please supply a slice. I.e. call drop(slice(-5,None))')
            else:
                self.df.drop(labels = self.df.index[labels], axis = 0, inplace = True)
        elif axis == 1:
            self.df.drop(labels, axis = 1, inplace=True)
        return self
    
    def select(self, rows: Union[int, List[int]] = None, colnames: Union[str, List[str]] = None):
        """
        Selects specified columns or rows (opposite of drop-method)

        Parameters
        ----------
        rows: Union[int, List[int]]
            The rows to be selected.
        colnames: Union[str, List[str]]
            The columns to be selected.
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> world_shapefile = (
        >>>        DataProcessingOrchestrator(name = 'world_shapefile')
        >>>        .import_data(filename = 'geospatial/shapefiles/international/world/world_highres.shp', directory = dir_data_raw)
        >>>        .select(colnames = ['ADMIN','geometry'])
        >>>        .rename_cols({'ADMIN': 'country'})
        >>>        .replace(colname = 'country', mapping_dict= map_naturalearth_countrynames)
        >>>        .filter(conditions = [('country',countries_in_the_world,'in')])
        >>>    )

        Notes
        -----
        Step-registration: yes        
        """
        vars            = locals()
        del vars['self']
        funcname        = 'select'
        self.register_step(funcname, vars) 

        colnames = self._check_for_col(colnames)
        if rows is not None and colnames is not None:
            selected_df = self.df.loc[rows, colnames]
        elif rows is not None:
            selected_df = self.df.loc[rows, :]
        elif colnames is not None:
            selected_df = self.df.loc[:, colnames]
        else:
            print('no selection took place')

        self.df = selected_df
        return self

    def rename_cols(self, colnames_dict: Dict[str, str]):
        """
        Rename columns

        Parameters
        ----------
        colnames_dict: Dict[str, str]
            Dictionary of current column name (keys) and new colum name (values)
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> world_shapefile = (
        >>>        DataProcessingOrchestrator(name = 'world_shapefile')
        >>>        .import_data(filename = 'geospatial/shapefiles/international/world/world_highres.shp', directory = dir_data_raw)
        >>>        .select(colnames = ['ADMIN','geometry'])
        >>>        .rename_cols({'ADMIN': 'country'})
        >>>    )

        Notes
        -----
        Step-registration: yes  
        """
        vars            = locals()
        del vars['self']
        funcname        = 'rename_cols'
        self.register_step(funcname, vars)   
        
        self.df = self.df.rename(columns=colnames_dict)
        return self

    def filter(self, conditions: List[Tuple[str, any, str]], logic: str = 'and'):
        """
        Filters on condition. The mask is created using dataprocessing.filtering.apply_condition

        Parameters
        ----------
        conditions: List[Tuple[str, any, str]]
            A list of tuples, each of which is a condition. A condition looks like:
                - colname
                - value
                - operator
        logic: str
            The logic behind dealing with multiple conditions ['and', 'or'].
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        --------
        >>> world_shapefile = (
        >>>        DataProcessingOrchestrator(name = 'world_shapefile')
        >>>        .import_data(filename = 'geospatial/shapefiles/international/world/world_highres.shp', directory = dir_data_raw)
        >>>        .select(colnames = ['ADMIN','geometry'])
        >>>        .rename_cols({'ADMIN': 'country'})
        >>>        .replace(colname = 'country', mapping_dict= map_naturalearth_countrynames)
        >>>        .filter(conditions = [('country',countries_in_the_world,'in')])
        >>>    )

        See also
        --------
        apply_condition in the filtering-subsubmodule --> creates a mask
        
        Notes
        -----
        Step-registration: yes  
        """
        vars            = locals()
        del vars['self']
        funcname        = 'filter'
     
        self.register_step(funcname, vars) 
        mask = apply_condition(self.df, conditions, logic)
        self.df = self.df[mask].reset_index(drop = True)
        return self

    def mutate(self, new_colname: str, value: Union[any, List[any], pd.Series, str] = None, operation: str = None, conditions: List[Tuple[str, any, str]] = None, logic: str = 'and'):
        """
        Diverse mutate function for columns.

        Parameters
        ----------
        new_colname: str
            The column name of the new column to be creates
        value: Union[any, List[any], pd.Series[any]]
            The value of the new column to be, if not row-dependent. It's therefore not possible to have both `value` and `operation` as not None
            When using a columnwise operation, list the old col_name as value!
        operation: str
            The string of the operation that includes the lambda functionality for rowwise operations. 
            For columnwise operations, please use functionality like "pd.to_datetime(x, format = "%b %Y")"
        conditions: List[Tuple[str, any, str]]
            A list of tuples, each of which is a condition. A condition looks like:
                - colname
                - value
                - operator
        logic: str
            The logic behind dealing with multiple conditions ['and', 'or'].
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        -------
        >>> example = ecdc_mv.copy()
        >>> example.mutate('newcol','1') # creates a new column with value '1' everywhere
        >>> example.mutate('newcol2', 'check', conditions = [('country', 'Austria', '==')]) # creates a new column with value 'check' everywhere for the given conditions
        >>> example.mutate('year', operation = "lambda row: int(str(row['year_month'])[:4])") # creates a new column with row-dependent values
        >>> example.mutate('newcol3', conditions = [('country', ['Austria', 'Belgium','Bulgaria'], 'in')], operation = "lambda row: row['country'] + row['country']") # Creates a new column with row-dependent values for given conditions.
        >>> iata_2012_01_06.mutate(new_colname = 'timestamp', value = "month", operation = "pd.to_datetime(x, format = '%b %Y')")
        >>> iata_2012_01_06.mutate(new_colname = 'factor', value = "proportion", operation = "x * 4")

        See also
        --------
        apply_condition in the filtering-subsubmodule --> creates a mask
        
        Notes
        -----
        Step-registration: yes         
        """
        vars            = locals()
        del vars['self']
        funcname        = 'mutate'
        self.register_step(funcname, vars)

        if new_colname not in self.df:
            self.df[new_colname] = None

        if conditions is None and value is not None and operation is None:
            self.df[new_colname] = value

        elif conditions is not None:
            mask = apply_condition(self.df, conditions, logic)
            if isinstance(operation, str) and operation.startswith('lambda'):
                self.df.loc[mask, new_colname] = self.df.loc[mask].apply(eval(operation), axis = 1)
            else:
                self.df.loc[mask, new_colname] = value
        
        elif conditions is None:
            # rowwise column operations 
            if isinstance(operation, str) and operation.startswith('lambda'):
                self.df[new_colname] = self.df.apply(eval(operation), axis = 1)

            # columnwise operation (operation based on another column)
            elif value:
                operation = operation.replace('x', f'self.df["{value}"]')

                self.df[new_colname] = eval(operation)

            else:
                raise ValueError(f'Error: either use rowwise operations (operation should start with "lambda") or columnwise operations.\nWhen using the latter one please supply the name of the column to base the operation on in "value"')
        return self

    def convert_to_geodataframe(self):
        """
        converts pd.DataFrame into gpd.GeoDataFrame with geometry colname 'geometry'.

        Parameters
        ----------
        None
            
        Returns
        -------
        DataProcessingOrchestrator
            Self with updated DataFrame   

        Examples
        -------
        >>> 
        
        Notes
        -----
        Step-registration: yes         
        """
        vars            = locals()
        del vars['self']
        funcname        = 'convert_to_geodataframe'
        self.register_step(funcname, vars)
        self.df = gpd.GeoDataFrame(self.df, geometry = 'geometry')

        return self

    def groupby(self, 
                groupby_columns: List[str], 
                aggregations: Dict[str, any]):
        """
        Groups the DataFrame based on specified columns and applies custom aggregations.
        
        Parameters
        ----------
        groupby_columns : List[str]
            List of columns to group by.
        aggregations : dict
            Dictionary where the keys are column names and the values are aggregation functions (or list of functions).

        Returns
        -------
        DataProcessor
            A new DataProcessor instance with the grouped and aggregated DataFrame.
        """
        vars = locals()
        del vars['self']
        funcname = 'groupby'

        self.register_step(funcname, vars)

        # Perform groupby operation and apply aggregations
        grouped_df = self.df.groupby(groupby_columns).agg(aggregations)

        # Return a new DataProcessor with the grouped and aggregated DataFrame
        self.df = grouped_df
        return self

    def reset_index(self, drop = False):
        """
       
        """
        vars = locals()
        del vars['self']
        funcname = 'reset_index'

        self.register_step(funcname, vars)
        self.df = self.df.reset_index(drop = drop)
        return self  

    def split_dfs(self, colname: str):
        df_dicts = {category: group for category, group in self.df.groupby(colname)}
        return df_dicts

    def merge_shapefile_world(self, on = 'country', how = 'outer'):
        from .df_merger import merge_world_shapefile
        """
       
        """
        vars = locals()
        del vars['self']
        funcname = 'merge_shapefile_world'

        self.register_step(funcname, vars)
        self.df = merge_world_shapefile(self.df, on = on, how = how)
        return self          
# Helpers - self
    def _check_for_col(self, column_names: Union[str, List[str]]):
        """
        helper function that checks for presence of column names in the dataframe.
        It also converts a column_name (str) into a list thereof, to be compatible with lists of multiple column names.

        Parameters
        ----------
        column_names: Union[str, List[str]]
            A column name or list thereof

        Returns
        -------
        column_names: List[str]
            A list of column_names that have been checked for appearance in the df.
        """
        column_names = _input_to_list(column_names)
        for col in column_names:
            if col not in self.df.columns:
                raise ValueError(f'{col} not a valid column name. Valid options are: {self.df.columns}')
        return column_names

# Helpers - nonself
def _input_to_list(input):
    """changes the input into a list of the input (only if it not already is, otherwise the input is returned)"""
    if not isinstance(input, list):
        input = [input]   
    return input