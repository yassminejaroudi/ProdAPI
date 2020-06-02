from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp
from regression_model.processing import features
from regression_model.config import config

import logging


_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        (
            "temporal_variable",
            pp.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS ),
        ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS),
        ),
        (
            "categorical_encoder",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "log_transformer",
            features.LogTransformer(variables=config.NUMERICALS_LOG_VARS),
        ),
        (
            "outlier_handler",
            pp.OutlierHandler(distance=3,variables=config.NUMERICALS_LOG_VARS),
        ),
         
        ("scaler", MinMaxScaler()),
        ('Linear_model', RandomForestRegressor(n_estimators=70, max_features='auto',min_samples_leaf= 2,
 min_samples_split= 2, random_state=0)),
    ]
)
