import warnings

# The following ones have to do with the fact that pyproj is not saved correctly into this environment. I believe this is because I have restricted access to the pc.
warnings.filterwarnings("ignore", message="pyproj unable to set database path.")
warnings.filterwarnings("ignore", message="Could not detect PROJ data files.*", category=RuntimeWarning)