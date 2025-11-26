"""
Configurações do projeto de previsão de demanda
"""

import os
from pathlib import Path

# Paths base
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Paths específicos
RAW_DATA_PATH = DATA_DIR / "raw" / "retail_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"
EXTERNAL_DATA_PATH = DATA_DIR / "external"
MODEL_PATH = MODELS_DIR / "demand_forecaster.pkl"

# Parâmetros do modelo
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# Parâmetros de negócio
BUSINESS_PARAMS = {
    'custo_estoque_excesso': 5,    # R$ por unidade/dia
    'custo_falta_estoque': 15,     # R$ por unidade perdida
    'margem_media': 0.3,           # 30% de margem
    'estoque_seguranca_percentual': 0.1  # 10% de estoque de segurança
}

# Configurações de feature engineering
FEATURE_PARAMS = {
    'lags': [1, 7, 30],
    'rolling_windows': [7, 30],
    'cyclic_features': ['mes', 'dia_ano']
}