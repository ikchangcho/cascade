import sys
import os
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from datetime import datetime
import importlib
import griess as gr
import bmgdata as bd
import denitfit as dn

# Load the pickle file
monocultures = pickle.load(open( "monocultures.pkl", "rb" ))

# Display the type of the loaded data
print(f'Type of data: {type(monocultures)}')

# Access the first object in the first sublist
first_experiment = monocultures[0][0]

# Get a list of all attributes and methods of the object
attributes = dir(first_experiment)
print("Attributes and methods of the first experiment object:")
for attribute in attributes:
    print(attribute)

# Get detailed information about the object
print("\nDetailed information about the first experiment object:")
help(first_experiment)
