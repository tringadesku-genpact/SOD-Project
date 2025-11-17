# train.py

from data_loader import debug_data_loader
from sod_model import debug_model

def main():
    print("train.py is running ✔")
    debug_data_loader()
    debug_model()

if __name__ == "__main__":
    main()
