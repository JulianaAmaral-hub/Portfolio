# Modelo com obtenção de dados de duas fontes distintas
# Análise de Medidas de Posição, Dispersão e Variabilidade

#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# Bibliotecas --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Funções: -----------------------------------------------------------------------------------------
from auxiliar.conexoes_trein import obter_dados

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
    
    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# FONTE DE *.csv: DP.csv ----------------------------------------------------------------------------

ACAO = 'Obter dados de DP'
try:
    print(f'\n{ACAO}...')
    df_dp = pd.read_csv('C:/BD SCIENCE/UC2/DADOS/DP.csv')
    print(df_dp.head())

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#____________________________________TRABALHANDO OS DADOS __________________________________________

# df_ocorrências: 
# cisp, roubo_veiculo, recuperacao_veiculos
# Período: mes_ano (2021 a 2023)

# df_dp
# Relação: nome e endereço por codDP

# Delimitar dataframe por período (2021 a 2023) e Variáveis ----------------------------------------
ACAO = 'Delimitar dataframe'
try:
    print(f'\n{ACAO}...')
    # Delimitar dataframe por período (2021 a 2023)
    df_ocorrencias_2021_2023 = df_ocorrencias[(df_ocorrencias ['ano'] >= 2021) & (df_ocorrencias ['ano'] <= 2023)]
    # Delimitar dataframe por variáveis (mes_ano, cisp, roubo_veiculo, recuperacao_veiculos)
    df_ocorrencias_2021_2023 = df_ocorrencias_2021_2023[['mes_ano', 'cisp', 'roubo_veiculo', 'recuperacao_veiculos']]
    #print(df_ocorrencias_2021_2023)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Totalizar roubo_veiculo, recuperacao_veiculos por cisp -------------------------------------------
ACAO = 'Totalizar dataframe'
try:
    print(f'\n{ACAO}...')
    df_ocorrencias_2021_2023_dp = df_ocorrencias_2021_2023.groupby('cisp').sum(['roubo_veiculo', 'recuperacao_veiculos']).reset_index()
    print(df_ocorrencias_2021_2023_dp)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Identificar Outliers de DP por roubo_veiculo e recuperacao_veiculos ------------------------------

ACAO = 'Identificar outliers'
try:
    print(f'\n{ACAO}...')

    # Arrays
    array_roubo_veiculo = np.array(df_ocorrencias_2021_2023_dp['roubo_veiculo'])
    array_recup_veiculo = np.array(df_ocorrencias_2021_2023_dp['recuperacao_veiculos'])
    # Q1, Q3 e IQR
    q1_roubo = np.quantile(array_roubo_veiculo, 0.25, method='weibull')
    q1_recup = np.quantile(array_recup_veiculo, 0.25, method='weibull')
    q3_roubo = np.quantile(array_roubo_veiculo, 0.75, method='weibull')
    q3_recup = np.quantile(array_recup_veiculo, 0.75, method='weibull')
    iqr_roubo = q3_roubo - q1_roubo
    iqr_recup = q3_recup - q1_recup
    # Limite Superior
    limite_sup_roubo = q3_roubo + (1.5 * iqr_roubo)
    limite_sup_recup = q3_recup + (1.5 * iqr_recup)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Relacionar dataframes  e filtrar outliers --------------------------------------------------------
# df_ocorrencias_2021_2023_dp('cisp') com df_dp('codDP') 
ACAO = 'Relacionar dataframes e filtrar outliers'
try:
    print(f'\n{ACAO}...')
    # Relacionar dataframes
    df_ocorrencias_2021_2023_nome_dp = pd.merge(df_ocorrencias_2021_2023_dp, df_dp, left_on='cisp', right_on='codDP')
    # Filtrar Outliers - roubo_veiculo
    df_dps_roubos = df_ocorrencias_2021_2023_nome_dp[df_ocorrencias_2021_2023_nome_dp['roubo_veiculo']>limite_sup_roubo]
    df_dps_roubos = df_dps_roubos[['nome', 'roubo_veiculo']]
    # Filtrar Outliers - recuperacao_veiculos
    df_dps_recup = df_ocorrencias_2021_2023_nome_dp[df_ocorrencias_2021_2023_nome_dp['roubo_veiculo']>limite_sup_recup]
    df_dps_recup = df_dps_recup[['nome', 'recuperacao_veiculos']]

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Consolidar por mes_ano e observar distribuição dos dados -----------------------------------------

ACAO = 'Observar meses e anos'
try:
    print(f'\n{ACAO}...')
    df_ocorrencias_2021_2023_mes_ano = df_ocorrencias_2021_2023[['mes_ano', 'roubo_veiculo', 'recuperacao_veiculos']].groupby('mes_ano').sum\
                                        (['roubo_veiculo', 'recuperacao_veiculos']).reset_index()
    print(df_ocorrencias_2021_2023_mes_ano)

    # Obter a assimetria da distribuição dos dados
    assimetria_roubo = df_ocorrencias_2021_2023_mes_ano['roubo_veiculo'].skew()
    assimetria_recup = df_ocorrencias_2021_2023_mes_ano['recuperacao_veiculos'].skew()
    # Média e Mediana
    media_roubo = np.mean(df_ocorrencias_2021_2023_mes_ano['roubo_veiculo'])
    media_recup = np.mean(df_ocorrencias_2021_2023_mes_ano['recuperacao_veiculos'])
    mediana_roubo = np.median(df_ocorrencias_2021_2023_mes_ano['roubo_veiculo'])
    mediana_recup = np.median(df_ocorrencias_2021_2023_mes_ano['recuperacao_veiculos'])
    print('')
    print(40*'=')
    print('\nAssimetrias:')
    print('Assimetria Roubo Veículo: ', assimetria_roubo)
    print('Média Roubo Veículo:', media_roubo)
    print('Mediana Roubo Veículo:', mediana_roubo)
    print('\nAssimetria Recuperação Veículos: ', assimetria_recup)
    print('Média Recuperação Veículos:', media_recup)
    print('Mediana Recuperação Veículos:', mediana_recup)
    print(40*'-')
    print('\nMedidas de Posição:')
    print('\nQ1 Roubo: ', q1_roubo)
    print('Q3 Roubo: ', q3_roubo)
    print('IQR Roubo: ', iqr_roubo)
    print('Limite Superior Roubo: ', limite_sup_roubo)  
    print('\nQ1 Recuperação Veículos: ', q1_recup)
    print('Q3 Recuperação Veículos: ', q3_recup)
    print('IQR Recuperação Veículos: ', iqr_recup)
    print('Limite Superior Recuperação Veículos: ', limite_sup_recup)
    print('')
    print(40*'=')

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Curtose -----------------------------------------------------------------------------------------

ACAO = 'Calcular a Curtose'
try:
    print(f'\n{ACAO}...')

    curtose_roubo = df_ocorrencias_2021_2023_mes_ano['roubo_veiculo'].kurtosis()
    curtose_recup = df_ocorrencias_2021_2023_mes_ano['recuperacao_veiculos'].kurtosis()
    print(40*'=')
    print('\nCurtose:')
    print('Curtose Roubo: ',curtose_roubo)
    print('Curtose Recuperação Veículo: ',curtose_recup)
    print(40*'=')

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# MEDIDAS DE DISPERSÃO OU VARIABILIDADE -----------------------------------------------------------

ACAO = 'Obter medidas de dispersão ou variabilidade'
try:
    print(f'\n{ACAO}...')

    # VARIÂNCIA:   
    var_roubo = np.var(array_roubo_veiculo)
    var_recup = np.var(array_recup_veiculo)
    dist_roubo = var_roubo/(media_roubo**2)
    dist_recup = var_recup/(media_recup**2)
    print(40*'=')
    print('\nMedidas de Dispersão:')
    print('Variância Roubo de Veículos: ', var_roubo)
    print('Distância Roubo de Veículos: ', dist_roubo)
    print('Variância Recuperação de Veículos: ', var_recup)
    print('Distância Recuperação de Veículos: ', dist_recup)
    # DESVIO PADRÃO:
    desvio_roubo = np.std(array_roubo_veiculo)
    desvio_recup = np.std(array_recup_veiculo)
    # COEFICIENTE DE VARIAÇÃO:
    coef_roubo = desvio_roubo/media_roubo
    coef_recup = desvio_recup/media_recup

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
    plt.subplots(2,2, figsize = (16,6))
    plt.suptitle('Análise de roubos e recuperações de veículos', fontsize=16)

    # Layout do Painel: 2 linhas e 1 colunas
    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 1
    plt.subplot(2,2,1)
    plt.title('Ranking de Delegacias com mais registros de roubos de veículos')
    df_dps_roubos_ordenado = df_dps_roubos.sort_values(by='roubo_veiculo').head(10)
    plt.barh(df_dps_roubos_ordenado['nome'], df_dps_roubos_ordenado['roubo_veiculo'], color = 'blue')

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 2
    plt.subplot(2,2,2)
    plt.title('As 10 DPs com mais registros de recuperação de veículos') 
    df_dps_recup_ordenado = df_dps_recup.sort_values(by='recuperacao_veiculos').head(10)
    plt.barh(df_dps_recup_ordenado['nome'], df_dps_recup_ordenado['recuperacao_veiculos'], color = 'orange')
    
    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 3
    plt.subplot(2,2,3)
    plt.title('Evolução de Roubos e Recuperação de Veículos')
    plt.bar(df_ocorrencias_2021_2023_mes_ano['mes_ano'], df_ocorrencias_2021_2023_mes_ano['roubo_veiculo'], color='blue', label = 'roubo_veiculo')  
    plt.bar(df_ocorrencias_2021_2023_mes_ano['mes_ano'], df_ocorrencias_2021_2023_mes_ano['recuperacao_veiculos'], color='orange', label = 'recuperacao_veiculos')
    plt.axhline(media_roubo, color='black', linestyle='--', label='Média Roubo de Veículos')
    plt.axhline(mediana_recup, color='red', linestyle='--', label='Média Recuperação de Veículos')
    plt.xticks(rotation = 90)    
    # Legenda:
    plt.legend(bbox_to_anchor=(1,0.35))

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 4
    plt.subplot(2,2,4)
    plt.text(0.1, 0.9375, f'Assimetria Roubo Veículos: {assimetria_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0.875, f'Média Roubo Veículos: {media_roubo}', fontsize=10, ha='left', color='blue')
    plt.text(0.1, 0.8125, f'Mediana Roubo Veículos: {mediana_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0.75, f'Curtose Roubo Veículos: {curtose_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0.6875, f'Assimetria Recuperação Veículos: {assimetria_recup}', fontsize=10, ha='left')
    plt.text(0.1, 0.625, f'Média Recuperação Veículos: {media_recup}', fontsize=10, ha='left', color='red')
    plt.text(0.1, 0.5625, f'Mediana Recuperação Veículos: {mediana_recup}', fontsize=10, ha='left')
    plt.text(0.1, 0.5, f'Curtose Recuperação Veículos: {curtose_recup}', fontsize=10, ha='left')
    
    plt.text(0.1, 0.4375, f'Variância Roubo Veículos: {var_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0.375, f'Variância Recuperação Veículos: {var_recup}', fontsize=10, ha='left')
    plt.text(0.1, 0.3125, f'Distância Roubo Veículos: {dist_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0.25, f'Distância Recuperação Veículos: {dist_recup}', fontsize=10, ha='left')
    plt.text(0.1, 0.1875, f'Desvio Padrão Roubo Veículos: {desvio_roubo}', fontsize=10, ha='left', color='blue')
    plt.text(0.1, 0.125, f'Desvio Padrão Recuperação Veículos: {desvio_recup}', fontsize=10, ha='left', color='red')
    plt.text(0.1, 0.0625, f'Coeficiente de Variação Roubo Veículos: {coef_roubo}', fontsize=10, ha='left')
    plt.text(0.1, 0, f'Coeficiente de Variação Recuperação Veículos: {coef_recup}', fontsize=10, ha='left')

    # ______________________________________________________________________________________________
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()