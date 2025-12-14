# data_mining
Data mining group project

Run one of these commands depending on your setup: replace {library name} with the library that you need to install

Example: 
pip install {library name}

The labraries used were the following: 
# Core libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: model selection & preprocessing
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
    KFold, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Scikit-learn: metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score, silhouette_score
)

# Scikit-learn: models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Scikit-learn: pipeline
from sklearn.pipeline import Pipeline

# XGBoost
from xgboost import XGBRegressor
