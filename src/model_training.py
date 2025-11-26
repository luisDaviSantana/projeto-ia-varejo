import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sys

# Adiciona o diretório pai ao path para importações relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Agora podemos importar corretamente
try:
    from src.data_processing import DataProcessor
except ImportError:
    # Fallback para desenvolvimento
    from data_processing import DataProcessor

class DemandForecaster:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.model_path = "models/demand_forecaster.pkl"
    
    def train(self, df, target_col='demanda'):
        """Treina o modelo de previsão"""
        # Definir features e target
        self.feature_columns = [col for col in df.columns 
                              if col not in ['data', target_col]]
        X = df[self.feature_columns]
        y = df[target_col]
        
        # Split temporal
        split_point = int(0.8 * len(df))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Treinar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Avaliar
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Score de treino: {train_score:.4f}")
        print(f"Score de teste: {test_score:.4f}")
        
        # Salvar modelo
        self._save()
        
        return self
    
    def predict(self, df):
        """Faz previsões"""
        if self.model is None:
            self._load()
        
        # Garantir todas as features necessárias
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns]
        return self.model.predict(X)
    
    def _save(self):
        """Salva o modelo internamente"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, self.model_path)
        print(f"Modelo salvo em: {self.model_path}")
    
    def _load(self):
        """Carrega o modelo internamente"""
        if os.path.exists(self.model_path):
            loaded = joblib.load(self.model_path)
            self.model = loaded['model']
            self.feature_columns = loaded['feature_columns']
            print("Modelo carregado com sucesso")
        else:
            raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")
        return self
    
    def load(self, path):
        """Carrega o modelo de um path específico (método público)"""
        if os.path.exists(path):
            loaded = joblib.load(path)
            self.model = loaded['model']
            self.feature_columns = loaded['feature_columns']
            print(f"Modelo carregado de: {path}")
        else:
            raise FileNotFoundError(f"Modelo não encontrado em: {path}")
        return self

# Exemplo de uso
if __name__ == "__main__":
    print("🚀 Iniciando treinamento do modelo...")
    
    try:
        # Tentar importar do módulo src
        from src.data_processing import DataProcessor
    except ImportError:
        print("⚠️  Importação do módulo src falhou. Usando import local...")
        # Criar uma versão simplificada para teste
        processor = DataProcessor()
    
    # Processar dados
    print("📊 Processando dados...")
    processor = DataProcessor()
    df = processor.load_and_process_data()
    df_features = processor.create_features(df)
    
    print(f"📁 Dados carregados: {len(df_features)} registros")
    print(f"🎯 Features criadas: {len(df_features.columns)} colunas")
    
    # Treinar modelo
    print("🤖 Treinando modelo...")
    forecaster = DemandForecaster()
    forecaster.train(df_features)
    
    print("✅ Modelo treinado e salvo com sucesso!")