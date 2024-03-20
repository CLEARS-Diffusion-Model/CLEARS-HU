# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 00:21:33 2022

@author: adh
"""

# Standard library imports
import copy
import os
import sys
import copy

wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(wd)

# Third party imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Local library imports
import source_code.paths_append
from model_class import ModelRun

print('Start')

# Instantiate the run
model = ModelRun()
# Fetch ModelRun attributes, for examination
# Titles of the model
titles = model.titles
# Dimensions of model variables
dims = model.dims
# Converters
converter = model.converter
# Data
data = model.data
d6_data = model.d6_data
# Timeline
timeline = model.timeline
# Set random seedő
np.random.seed(123)
# Run model
model.run()

# Export results
results = copy.deepcopy(model.data)
d6_results = copy.deepcopy(model.d6_data)
print('Model run finished, exporting results')

run_id = model.name
with open('output\{}.pickle'.format(run_id), 'wb') as f:
    pickle.dump(model,f)

print('Results are exported')

