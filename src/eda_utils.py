import pandas as pd
import numpy as np


color_palette = ['#023047', '#e85d04', '#0077b6', '#ff8200', '#0096c7', '#ff9c33']
PALETTE_MMM = {
    'Vendas': color_palette[1],  
    'Investimento_TV': color_palette[0], 
    'Investimento_Google_Ads': color_palette[2], 
    'Investimento_Facebook_Ads': color_palette[4], 
    'Fator_Feriado': color_palette[3],
    'Fator_Sazonalidade': color_palette[5]
}

def gerar_dicionario_e_auditoria(df):
    """
    Realiza uma auditoria de qualidade na série temporal e gera um dicionário de dados.
    Focado nas premissas rigorosas do Marketing Mix Modeling.
    """
    print("🕒 --- AUDITORIA DE CONTINUIDADE TEMPORAL ---")
    
    # Verifica se o índice é temporal
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("❌ CRÍTICO: O índice do DataFrame não é do tipo Datetime. Modelos de Adstock falharão.")
    else:
        # Calcula a diferença de dias entre as linhas
        dias_diff = df.index.to_series().diff().dt.days.dropna()
        
        # O LightweightMMM e a maioria dos modelos assumem dados semanais (7 dias exatos)
        if (dias_diff == 7).all():
            print(f"✅ Série temporal íntegra: {len(df)} semanas consecutivas com espaçamento perfeito de 7 dias.")
        else:
            print("⚠️ ALERTA DE RUIDO TEMPORAL: Gaps ou espaçamentos irregulares detectados!")
            print(f"Padrões de espaçamento encontrados (dias): \n{dias_diff.value_counts()}")
            
    print("\n📊 --- DICIONÁRIO DE DADOS E TIPAGEM ---")
    dicionario = pd.DataFrame({
        'Tipo': df.dtypes,
        'Nulos (Qtd)': df.isnull().sum(),
        'Nulos (%)': (df.isnull().sum() / len(df) * 100).round(2),
        'Valores Únicos': df.nunique(),
        'Exemplo (1ª Linha)': df.iloc[0].values
    })
    
    return dicionario

def analise_descritiva_mmm(df):
    """Gera estatísticas descritivas com foco em assimetria para MMM."""
    desc = df.describe().T
    desc['assimetria (skew)'] = df.skew()
    return desc[['mean', 'std', 'min', '50%', 'max', 'assimetria (skew)']].round(2)



import matplotlib.pyplot as plt

def plot_temporal_trends(df, colors=PALETTE_MMM):
    """Gera o gráfico de linhas temporal com a paleta oficial do projeto."""
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Eixo 1: Investimentos (Tons de Azul da sua paleta)
    ax1.plot(df.index, df['Investimento_TV'], label='TV', color=colors['Investimento_TV'], linewidth=1.5)
    ax1.plot(df.index, df['Investimento_Google_Ads'], label='Google Ads', color=colors['Investimento_Google_Ads'], linewidth=1.5)
    ax1.plot(df.index, df['Investimento_Facebook_Ads'], label='Facebook Ads', color=colors['Investimento_Facebook_Ads'], linewidth=1.5)
    
    ax1.set_ylabel('Investimento (R$)', fontweight='bold')
    ax1.set_title('Dinâmica Temporal: Canais de Mídia vs Vendas', loc='left', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.2)

    # Eixo 2: Vendas (Destaque em Laranja da sua paleta)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Vendas'], color=colors['Vendas'], linewidth=3, label='Vendas (Target)', linestyle='-')
    ax2.set_ylabel('Vendas (Volume)', fontweight='bold', color=colors['Vendas'])
    ax2.tick_params(axis='y', labelcolor=colors['Vendas'])
    ax2.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    plt.show()


import seaborn as sns

def plot_boxplots_marketing(df, colors=PALETTE_MMM):
    """Gera boxplots individuais para detecção de outliers com a paleta oficial."""
    cols = ['Investimento_TV', 'Investimento_Google_Ads', 'Investimento_Facebook_Ads', 'Vendas']
    custom_pal = [colors[c] for c in cols]
    
    plt.figure(figsize=(16, 5))
    sns.boxplot(data=df[cols], palette=custom_pal, orient='h')
    
    plt.title('Distribuição e Dispersão (Outliers)', loc='left', fontsize=14, fontweight='bold')
    plt.xlabel('Valor')
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, colors=PALETTE_MMM):
    """Gera um heatmap de correlação usando a paleta oficial."""
    plt.figure(figsize=(10, 8))
    
    # Criando uma colormap baseada na sua paleta (Azul para negativo, Laranja para positivo)
    cmap = sns.diverging_palette(240, 30, as_cmap=True) # Ajuste visual para combinar com seus tons
    
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Matriz de Correlação: Mídia, Controles e Target', loc='left', fontsize=14, fontweight='bold')
    plt.show()




def plot_lagged_correlation(df, target='Vendas', max_lags=4, colors=PALETTE_MMM):
    """Analisa a correlação de mídia com o target em diferentes janelas de tempo (Lags)."""
    media_cols = ['Investimento_TV', 'Investimento_Google_Ads', 'Investimento_Facebook_Ads']
    lag_data = []

    for col in media_cols:
        for lag in range(max_lags + 1):
            correlation = df[target].corr(df[col].shift(lag))
            lag_data.append({'Canal': col, 'Lag (Semanas)': lag, 'Correlação': correlation})

    df_lag = pd.DataFrame(lag_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_lag, x='Lag (Semanas)', y='Correlação', hue='Canal', 
                palette=[colors[c] for c in media_cols])
    
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title('Análise de Impacto Residual (Lagged Correlation)', loc='left', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    plt.show()


def plot_numerical_distributions(df, colors=PALETTE_MMM):
    """Gera histogramas com curvas de densidade (KDE) para variáveis chave."""
    cols = ['Investimento_TV', 'Investimento_Google_Ads', 'Investimento_Facebook_Ads', 'Vendas']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors.get(col, '#000000'), bins=20)
        axes[i].set_title(f'Distribuição: {col}', fontweight='bold')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')

    sns.despine()
    plt.tight_layout()
    plt.show()