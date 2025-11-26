import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataProcessor:
    def __init__(self):
        self.data_path = "data/raw/retail_data.csv"
        self.processed_path = "data/processed/processed_data.csv"

    def generate_sample_data(self):
        """Gera dados de exemplo se não existirem"""
        if not os.path.exists(self.data_path):
            print("📁 Gerando dados de exemplo...")
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

            dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
            np.random.seed(42)

            data = []
            for date in dates:
                seasonal = 100 + 50 * np.sin(2 * np.pi * date.dayofyear / 365)
                trend = 0.1 * (date - dates[0]).days
                weekday_effect = [0, -10, -5, 0, 5, 15, 20][date.weekday()]

                special_event = 0
                if date.month == 12 and date.day <= 25:
                    special_event = 80
                elif date.month == 6:
                    special_event = 40

                base_demand = seasonal + trend + weekday_effect + special_event
                demand = max(0, base_demand + np.random.normal(0, 15))

                data.append({
                    'data': date,
                    'demanda': demand,
                    'preco_medio': np.random.uniform(50, 150),
                    'promocao': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'feriado': 1 if special_event > 0 else 0,
                    'temperatura': np.random.normal(25, 8)
                })

            df = pd.DataFrame(data)
            df.to_csv(self.data_path, index=False)
            print(f"✅ Dados gerados em: {self.data_path}")
            print(f"📊 Período: {df['data'].min()} até {df['data'].max()}")
        else:
            print(f"✅ Dados já existem em: {self.data_path}")

    def load_and_process_data(self):
        """Carrega e processa os dados"""
        self.generate_sample_data()

        df = pd.read_csv(self.data_path)
        df['data'] = pd.to_datetime(df['data'])

        # Salvar dados processados
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)

        print(f"📈 Dados processados: {len(df)} registros")
        return df

    def create_features(self, df):
        """Cria features para o modelo"""
        print("🔧 Criando features...")
        df = df.copy()

        # Features temporais
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
        df['dia_ano'] = df['data'].dt.dayofyear
        df['semana_ano'] = df['data'].dt.isocalendar().week
        df['trimestre'] = df['data'].dt.quarter
        df['final_semana'] = (df['data'].dt.dayofweek >= 5).astype(int)

        # Features cíclicas
        df['mes_sin'] = np.sin(2 * np.pi * df['mes']/12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes']/12)
        df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano']/365)
        df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano']/365)

        # Lags
        for lag in [1, 7, 30]:
            df[f'demanda_lag_{lag}'] = df['demanda'].shift(lag)

        # Médias móveis
        for window in [7, 30]:
            df[f'demanda_media_{window}'] = df['demanda'].rolling(
                window).mean()

        result = df.dropna()
        print(f"✅ Features criadas: {len(result.columns)} colunas totais")
        return result
