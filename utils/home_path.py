
from pathlib import Path
import sys 

def set_path():
    sys.path.append(str(Path(__file__).resolve().parent.parent))



def __main__():
    set_path()