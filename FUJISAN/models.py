"""FUJISAN model registry: model files, feature schemas, loading and prediction.

Kept free of heavy imports (scikit-learn, matplotlib, SHAP) so that the
feature-generation CLI (``fujisan``) can load a model quickly. Moved here
unchanged from run_FUJISAN.py, which imports these names from this module.
"""

import os

import lightgbm as lgb


def _model_dir():
    """Directory of the model files.

    Installed package (wheel): ``FUJISAN/model`` inside the package, found with
    importlib.resources (the files are copied there at build time, see
    pyproject.toml). Source checkout / editable install: ``model/`` at the
    repository root.
    """
    try:
        from importlib.resources import files
        packaged = files('FUJISAN') / 'model'
        if packaged.is_dir():
            return str(packaged)
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')


MODEL_DIR = _model_dir()

# Classification threshold of the published model (Fujita & Terada, 2024).
# Fixed value; it is NOT optimized on the test set.
PAPER2024_THRESHOLD = 0.5

PAPER2024_FEATURES = ['Full_log10(E_value)', 'Full_SeqID', 'Dmn_RMSD', 'Dmn_SeqID', 'Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID']
UPDATED_FEATURES = ['BitScore', 'Full_SeqID', 'Dmn_RMSD', 'Dmn_SeqID', 'Dmn_TM']

MODEL_CONFIGS = {
    'paper2024': {
        'model_path': os.path.join(MODEL_DIR, 'fujisan_paper2024_evalue_pocket_lgbm.txt'),
        'features': PAPER2024_FEATURES,
        'threshold': PAPER2024_THRESHOLD,
        'description': 'Model used in the published study (E-value + sequence + domain + pocket features)',
    },
    'updated': {
        'model_path': os.path.join(MODEL_DIR, 'fujisan_updated_blastbitscore_nopocket_lgbm.txt'),
        'features': UPDATED_FEATURES,
        # No validated threshold is available for this model in this repository.
        'threshold': None,
        'description': 'Post-publication model (BitScore + sequence + domain features, no pocket features)',
    },
}


def load_model(model_name):
    """
    Load a FUJISAN model and check that its feature schema matches MODEL_CONFIGS
    """
    config = MODEL_CONFIGS[model_name]
    if not os.path.isfile(config['model_path']):
        raise FileNotFoundError('Model file not found: ' + config['model_path'])
    model = lgb.Booster(model_file=config['model_path'])
    if model.feature_name() != config['features']:
        raise ValueError(
            "Feature schema of model '" + model_name + "' does not match MODEL_CONFIGS.\n"
            f"Model file: {model.feature_name()}\n"
            f"MODEL_CONFIGS: {config['features']}"
        )
    return model


def check_features(model, X):
    """
    Raise an error if the columns of X differ from the features expected by the model
    """
    expected_features = model.feature_name()
    if list(X.columns) != expected_features:
        raise ValueError(
            "Feature mismatch between the selected model and input data.\n"
            f"Model expects: {expected_features}\n"
            f"Input contains: {list(X.columns)}"
        )


def predict_proba(model, X):
    """
    Predict the probability that a protein pair has identical function
    """
    check_features(model, X)
    return model.predict(X, num_iteration=model.best_iteration)
