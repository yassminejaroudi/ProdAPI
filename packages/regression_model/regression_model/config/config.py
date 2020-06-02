import pathlib

import regression_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = 'Valeur_fonciere'

TEMPORAL_VARS = 'Date_mutation'

# input variables 
FEATURES = ['Date_mutation', 'Nature_mutation', 'No_voie',
       'Type_de_voie', 'Voie', 'Commune', 'Code_departement',
        'Section', 'Type_local', 'Surface_reelle_bati',
       'Nombre_pieces_principales', 'Nature_culture', 'Surface_terrain']


# categorical variables to encode
CATEGORICAL_VARS = ['Nature_mutation','Type_de_voie','Voie',
                    'Commune','Code_departement',
                     'Section', 'Type_local',
                    'Nature_culture']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['Type_de_voie','Voie',
                            'Section', 'Type_local','Nature_culture']


# numerical variables
NUMERICAL_VARS = ['No_voie','Surface_reelle_bati',
                'Nombre_pieces_principales','Surface_terrain']


# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['No_voie','Surface_reelle_bati',
                        'Nombre_pieces_principales','Surface_terrain']


# variables to log transform
NUMERICALS_LOG_VARS =  ['Surface_reelle_bati', 'Surface_terrain']

TYPES = {"Date_mutation" : "object",
            "Nature_mutation" : "object",
            "Valeur_fonciere" : "float64",
            "No_voie" : "float64",
            "Type_de_voie" : "object",
            "Voie" : "object",
            "Code_postal" : "float64",
            "Commune" : "object",
            "Code_departement" : "object",
            "Code_commune" : "int64",
            "Section" : "object",
            "Type_local" : "object",
            "Surface_reelle_bati" : "float64",
            "Nombre_pieces_principales" : "float64",
            "Nature_culture" : "object",
            "Surface_terrain" : "float64"}


NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]


PIPELINE_NAME = "randomforest_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
