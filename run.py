import warnings
import os
import sys

# Maximum warning suppression
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['PYTHONWARNINGS'] = 'ignore'

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Now import and run main
if __name__ == "__main__":
    import main
    main.main()