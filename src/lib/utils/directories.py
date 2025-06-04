import os
from dotenv import load_dotenv

load_dotenv()
#####################################################################################################
# Get base project directory from environment variable ##############################################
dir_project = os.getenv('PROJECT_ROOT') 

dir_data            = os.path.join(dir_project,'data')
dir_data_raw        = os.path.join(dir_data,'raw')
dir_data_processed  = os.path.join(dir_data,'processed')