# PROJETO: UC2 - EXEMPLO 11 Detecção de Anomalias
# Modelo com análise de dados em larga escala.
# Detecção de anomalias no pagamento das parcelas do bolsa família nos meses de janeiro a maio de 2024.
# Arquivo de cálculo
#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# Bibliotecas --------------------------------------------------------------------------------------
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
# Funções: -----------------------------------------------------------------------------------------
from datetime import datetime 
from sklearn.ensemble import IsolationForest 

#___________________________________________________________________________________________________
#________________________________________OBTENDO DADOS______________________________________________

# FONTE DE DADOS: ----------------------------------------------------------------------------------
# Novo Bolsa Família
# Exercício 2024, meses Janeiro a Maio
# ARQUIVO PARQUET

# CONSTANTES:
ENDERECO_DADOS = r'./DADOS/'

ACAO = 'Obter dados arquivo parquet'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')
    df_bf = pl.read_parquet(ENDERECO_DADOS + 'dados_df_bf_2024.parquet')  
    # Substituir Vírgula por Ponto e transformar os dados da coluna VALOR PARCELA em float
    df_bf = df_bf.with_columns(pl.col('VALOR PARCELA').str.replace(',','.').cast(pl.Float64))
    print(df_bf.head())

    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#____________________________________TRABALHANDO OS DADOS __________________________________________

# Detecção de Anomalias  ---------------------------------------------------------------------------

ACAO = 'Detectar Anomalias'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')

    with pl.StringCache(): 
        # Array de VALOR PARCELA
        array_valor_parcela = np.array(df_bf.select('VALOR PARCELA'))
        # Definir o modelo do Isolation Forest 
        modelo = IsolationForest(contamination=0.005, random_state=42, n_estimators=100, n_jobs=1)
        # Aplicar o Isolation Forest no array valor parcela
        anomalias = modelo.fit_predict(array_valor_parcela.reshape(-1,1)) 
        # Adicionar as anomalias no df_bf
        df_bf_lazy = df_bf.lazy()
        df_bf = df_bf_lazy.with_columns([pl.Series(name='anomalia', values=anomalias)]) 
        df_bf = df_bf.collect()
        # Filtrar anômalos.
        df_bf_lazy = df_bf.lazy()
        df_bf = df_bf_lazy.filter(pl.col('anomalia') == -1) 
        df_anomalias_detectadas = df_bf = df_bf.collect()

    print('\nAnomalias identificadas:')
    print(df_anomalias_detectadas.sort('VALOR PARCELA', descending=True).head(20))

    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#___________________________________VISUALIZANDO OS DADOS __________________________________________

ACAO = 'Visualizar dados'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')

    # ______________________________________________________________________________________________
    # ------------------------------------------------------------------------------Título do Painel
    plt.subplots(2,2, figsize = (17,7))
    plt.suptitle('Detecção de anomalias no valor parcela', fontsize=16)

    # Layout do Painel: 2 linhas e 2 colunas
    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 1
    # Distribuição das parcelas anômalas
    array_valor_parcela_anomalia = np.array(df_anomalias_detectadas.select('VALOR PARCELA'))
    plt.subplot(2,2,1)
    plt.title('Distribuição das parcelas anômalas')
    plt.boxplot(array_valor_parcela_anomalia, vert=False, showmeans=True)

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 2
    # Histograma das parcelas anômalas
    plt.subplot(2,2,2)
    plt.title('Histograma das parcelas anômalas')
    plt.hist(array_valor_parcela_anomalia, bins=100, edgecolor='black')

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 3
    # Ranking das parcelas anômalas: 10 maiores parcelas
    plt.subplot(2,2,3)
    plt.title('Ranking das parcelas anômalas: 10 maiores parcelas')
    df_anomalias_detectadas_maiores = df_anomalias_detectadas.sort('VALOR PARCELA', descending=True).head(10)
    colunas = ['CPF FAVORECIDO', 'MÊS COMPETÊNCIA', 'MÊS REFERÊNCIA', 'UF', 'NOME MUNICÍPIO', 'VALOR PARCELA']
    x=0.1
    y=0.9
    for cpf, mes_competencia, mes_referencia, uf, municipio, valor_parcela in df_anomalias_detectadas_maiores[colunas].to_pandas().values:
        plt.text(x,y,f'{cpf}-{mes_competencia}-{mes_referencia}-{uf}-{municipio}-R${valor_parcela: .2f}', fontsize=8)
        y-=0.1
    plt.axis('off')

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 4
    # Ranking das parcelas anômalas: 10 menores parcelas
    plt.subplot(2,2,4)
    plt.title('Ranking das parcelas anômalas: 10 menores parcelas')
    df_anomalias_detectadas_menores = df_anomalias_detectadas.sort('VALOR PARCELA', descending=False).head(10)
    x=0.1
    y=0.9
    for cpf, mes_competencia, mes_referencia, uf, municipio, valor_parcela in df_anomalias_detectadas_menores[colunas].to_pandas().values:
        plt.text(x,y,f'{cpf}-{mes_competencia}-{mes_referencia}-{uf}-{municipio}-R${valor_parcela: .2f}', fontsize=8)
        y-=0.1 
    plt.axis('off')

    # ______________________________________________________________________________________________
    plt.tight_layout() 
    
    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)

    plt.show()

except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()