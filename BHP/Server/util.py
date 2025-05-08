
import pickle
import json
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    
    try:
        # Case-insensitive match
        loc_index = next((i for i, col in enumerate(__data_columns) if col.lower() == location.lower()), -1)
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)



def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(base_dir, "artifacts", "columns.json")
    model_path = os.path.join(base_dir, "artifacts", "banglore_home_prices_model.pickle")

    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    model_path = os.path.join(base_dir, "artifacts", "banglore_home_prices_model.pickle")
    with open(model_path, 'rb') as f:
        __model = pickle.load(f)


    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':

    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('ejipura', 1000, 2, 2))  # other location