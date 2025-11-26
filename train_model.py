# train_model.py na RAIZ do projeto
from src.model_training import DemandForecaster
from src.data_processing import DataProcessor
import sys
import os

# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("🚀 Iniciando treinamento do modelo...")

    # Processar dados
    print("📊 Processando dados...")
    processor = DataProcessor()
    df = processor.load_and_process_data()
    df_features = processor.create_features(df)

    print(f"📁 Dados carregados: {len(df_features)} registros")

    # Treinar modelo
    print("🤖 Treinando modelo...")
    forecaster = DemandForecaster()
    forecaster.train(df_features)

    print("✅ Modelo treinado e salvo com sucesso!")


if __name__ == "__main__":
    main()
