import pandas as pd
import numpy as np


class BusinessImpactCalculator:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, df, predictions):
        """Calcula métricas de impacto de negócio"""

        # Erro de previsão
        mae = np.mean(np.abs(df['demanda'] - predictions))
        mape = np.mean(
            np.abs((df['demanda'] - predictions) / df['demanda'])) * 100

        # Simular impacto financeiro
        custo_estoque_excesso = 5  # R$ por unidade/dia
        custo_falta_estoque = 15   # R$ por unidade perdida

        estoque_ideal = predictions
        excesso = np.maximum(estoque_ideal - df['demanda'], 0)
        falta = np.maximum(df['demanda'] - estoque_ideal, 0)

        custo_total_excesso = np.sum(excesso) * custo_estoque_excesso
        custo_total_falta = np.sum(falta) * custo_falta_estoque
        custo_total_antigo = custo_total_excesso + custo_total_falta

        # Com modelo otimizado (estoque = previsão + 10% segurança)
        estoque_otimizado = predictions * 1.1
        excesso_otimizado = np.maximum(estoque_otimizado - df['demanda'], 0)
        falta_otimizado = np.maximum(df['demanda'] - estoque_otimizado, 0)

        custo_excesso_otimizado = np.sum(
            excesso_otimizado) * custo_estoque_excesso
        custo_falta_otimizado = np.sum(falta_otimizado) * custo_falta_estoque
        custo_total_otimizado = custo_excesso_otimizado + custo_falta_otimizado

        economia = custo_total_antigo - custo_total_otimizado
        reducao_percentual = (economia / custo_total_antigo) * \
            100 if custo_total_antigo > 0 else 0

        self.metrics = {
            'mae': mae,
            'mape': mape,
            'custo_total_antigo': custo_total_antigo,
            'custo_total_otimizado': custo_total_otimizado,
            'economia_total': economia,
            'reducao_percentual': reducao_percentual,
            'custo_excesso_antigo': custo_total_excesso,
            'custo_falta_antigo': custo_total_falta,
            'custo_excesso_otimizado': custo_excesso_otimizado,
            'custo_falta_otimizado': custo_falta_otimizado
        }

        return self.metrics

    def generate_report(self):
        """Gera relatório executivo"""

        report = f"""
        RELATÓRIO DE IMPACTO - SISTEMA DE PREVISÃO DE DEMANDA
        =====================================================
        
        📊 MÉTRICAS DE PERFORMANCE:
        - Erro Absoluto Médio (MAE): {self.metrics['mae']:.1f} unidades
        - Erro Percentual Médio (MAPE): {self.metrics['mape']:.1f}%
        
        💰 IMPACTO FINANCEIRO:
        - Custo anterior (sem otimização): R$ {self.metrics['custo_total_antigo']:,.0f}
        - Custo atual (com otimização): R$ {self.metrics['custo_total_otimizado']:,.0f}
        - Economia total: R$ {self.metrics['economia_total']:,.0f}
        - Redução de custos: {self.metrics['reducao_percentual']:.1f}%
        
        🎯 BENEFÍCIOS ESPECÍFICOS:
        - Redução custo excesso estoque: {((self.metrics['custo_excesso_antigo'] - self.metrics['custo_excesso_otimizado']) / self.metrics['custo_excesso_antigo'] * 100):.1f}%
        - Redução custo falta estoque: {((self.metrics['custo_falta_antigo'] - self.metrics['custo_falta_otimizado']) / self.metrics['custo_falta_antigo'] * 100):.1f}%
        
        💡 RECOMENDAÇÕES:
        - Implementar sistema em todas as categorias
        - Revisar políticas de reposição
        - Monitorar sazonalidades específicas
        """

        return report
