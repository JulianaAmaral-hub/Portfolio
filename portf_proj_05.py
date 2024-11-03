# Clusterização
# Clusterização dos registros de roubo frente a recuperação de veículos,
# com base nos dados do Instituto de Segurança Pública do Estado do Rio de Janeiro.

#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# Bibliotecas --------------------------------------------------------------------------------------
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
# Funções: -----------------------------------------------------------------------------------------
from datetime import datetime 
from auxiliar.conexoes import obter_dados
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 

#___________________________________________________________________________________________________
#________________________________________OBTENDO DADOS______________________________________________

# FONTE DE DADOS WEB: site ISPRJ -------------------------------------------------------------------

# Instituto de Segurança Pública do Estado do Rio de Janeiro
# Estatística de segurança: série histórica mensal por área de delegacia desde 01/2003
# CONSTANTES:
ENDERECO_DADOS = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

ACAO = 'Obter dados de ocorrências'
try:
    print(f'\n{ACAO}...')
    df_ocorrencias = obter_dados(ENDERECO_DADOS,'','csv',';')
    print(df_ocorrencias.head())

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#____________________________________TRABALHANDO OS DADOS __________________________________________

# df_veículos: 
# cisp, roubo_veiculo, recuperacao_veiculos

# Delimitar dataframe por variáveis e totalizar por DP ---------------------------------------------
ACAO = 'Delimitar dataframe'
try:
    print(f'\n{ACAO}...')
    # Delimitar dataframe por variáveis (cisp, roubo_veiculo, recuperacao_veiculos)
    df_veiculos = df_ocorrencias[['cisp', 'roubo_veiculo', 'recuperacao_veiculos']]
    df_total_veiculos = df_veiculos.groupby('cisp').sum(['roubo_veiculo', 'recuperacao_veiculos']).reset_index()
    print(df_total_veiculos.head())

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Normalizar dados ---------------------------------------------------------------------------------
ACAO = 'Normalizar a escala dos dados'
try:
    print(f'\n{ACAO}...')
    # Arrays: Variáveis de entrada (X)
    array_roubo_veiculo = np.array(df_total_veiculos['roubo_veiculo'])
    array_recup_veiculos = np.array(df_total_veiculos['recuperacao_veiculos'])
    X = np.column_stack([array_roubo_veiculo, array_recup_veiculos])
    # Normalizar a escala
    scaler = StandardScaler()
    array_normalizado = scaler.fit_transform(X)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Método de Cotovelo: identificar a quantidade de clusters -----------------------------------------
ACAO = 'Normalizar a escala dos dados'
try:
    print(f'\n{ACAO}...')
    # importar a classe Kmeans    
    inercia = [] 
    valores_k = range(1,10)
    # Aplicar o método de cotovelo
    for k in valores_k:
        kmeans = KMeans(n_clusters=k, random_state=42) 
        # treinar o modelo com os dados normalizados
        kmeans.fit(array_normalizado)
        # adicionar a inercia à lista de inércias
        inercia.append(kmeans.inertia_)
    # visualizar a sugestão de quantidade de clusters
    plt.plot(valores_k,inercia)
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inércia')
    plt.title('Método de Cotovelo')
    plt.show()

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Clusterizar --------------------------------------------------------------------------------------
# Sendo: k = 4

ACAO = 'Clusterizar'
try:
    print(f'\n{ACAO}...')
    # Iniciar o modelo com k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    # Treinar o modelo (dividir os clusters)
    kmeans.fit(array_normalizado)
    # Adicionar o dataframe aos clusters
    df_total_veiculos['Cluster'] = kmeans.labels_
    df_total_veiculos = df_total_veiculos.assign(cluster=kmeans.labels_)
    pd.set_option('display.max_rows', None)
    print(df_total_veiculos)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#___________________________________VISUALIZANDO OS DADOS __________________________________________

ACAO = 'Visualizar dados'
try:
    print(f'\n{ACAO}...')
    # ______________________________________________________________________________________________
    # ------------------------------------------------------------------------------Título do Painel
    plt.subplots(1,1, figsize = (17,7))
    plt.suptitle('Clusterização de Roubo Veículo x Recuperação Veículos, por delegacias', fontsize=16)
    # Layout do Painel: 1 linhas e 1 colunas
    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 1
    plt.subplot(1,1,1)
    array_cluster = np.array(df_total_veiculos['Cluster'])
    scatter = plt.scatter(array_roubo_veiculo, array_recup_veiculos, c=array_cluster, cmap='plasma')
    plt.title('Clusters de Roubo Veículo x Recuperação Veículos')
    plt.xlabel('Roubo Veículo')
    plt.ylabel('Recuperação Veículos')
    # Legenda de cor para os clusters:
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(np.unique(array_cluster))
    cbar.set_label('Cluster')
    plt.ticklabel_format(style='plain', axis='both')
    # ______________________________________________________________________________________________
    plt.tight_layout()
    plt.show()

    print(f'Comando ({ACAO}) executado com sucesso!')
     
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()