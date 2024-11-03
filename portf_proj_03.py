# Regressão Linear, Análise Preditiva
# Com base nos dados de roubos e recuperação de veículos do Instituto de Segurança Pública 
# do Estado do Rio de Janeiro, prever qual seria a quantidade de veículos recuperados, 
# caso de no número de roubos forem 400.000, 500.000 e 600.000 veículos respectivamente nos
# próximos 3 meses

#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# Bibliotecas --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Funções: -----------------------------------------------------------------------------------------
from auxiliar.conexoes_trein import obter_dados
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
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

# Delimitar dataframe por municípios, delegacias, roubos e recuperações de veículos ----------------

ACAO = 'Delimitar de variáveis e totalização'
try:
    print(f'\n{ACAO}...')

    # Delimitar variáveis
    df_veiculos = df_ocorrencias[['munic', 'cisp', 'roubo_veiculo','recuperacao_veiculos']]
    # Totalizar dataframe por delegacias
    df_total_veiculos = df_veiculos.groupby('cisp').sum(['roubo_veiculo','recuperacao_veiculos']).reset_index()
    print(df_total_veiculos)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Correlação dos dados -----------------------------------------------------------------------------
# Ajuste de modelo:
# Série roubo_veiculo com valores muito dispersos. Cortar 5% dos valores maiores.
# Reduzir série recuperacao_veiculos para equilibar com o corte de roubo_veiculo.

ACAO = 'Analisar os dados'
try:
    print(f'\n{ACAO}...')
    # roubo_veiculo - 5% dos valores maiores
    df_total_veiculos_cut = df_total_veiculos[df_total_veiculos['roubo_veiculo'] < np.percentile(df_total_veiculos['roubo_veiculo'],95)]
    # recuperacao_veiculos - 1% dos valores maiores
    df_total_veiculos_cut = df_total_veiculos_cut[df_total_veiculos_cut['recuperacao_veiculos'] < np.percentile(df_total_veiculos_cut['roubo_veiculo'],95)]
    # Array das séries
    array_roubo_veiculo = np.array(df_total_veiculos_cut['roubo_veiculo'])
    array_recuperacao_veiculos = np.array(df_total_veiculos_cut['recuperacao_veiculos'])
    # Correlação entre as variáveis
    correlacao = np.corrcoef(array_roubo_veiculo, array_recuperacao_veiculos)[0,1]

    print('Correlação: ', correlacao)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Análise Preditiva: Regressão Linear ---------------------------------------------------------------

ACAO = 'Criar modelo para análise preditiva'
try:
    print(f'\n{ACAO}...')

    # Dividir os dados de treino e teste ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(array_roubo_veiculo, array_recuperacao_veiculos, 
        test_size=0.25, random_state=42)
    # Importar a classe de normalização -------------------------------------------------------------
    scaler = StandardScaler()
    # Normalização dos dados de Roubo de Veículos (X)
    X_train = scaler.fit_transform(X_train.reshape(-1,1))
    # Dados de teste (X_test)
    X_test = scaler.transform(X_test.reshape(-1,1))
    # Criar o modelo linear -----------------------------------------------------------------------
    modelo = LinearRegression()
    # Treinar o modelo com os dados de treino -----------------------------------------------------
    modelo.fit(X_train, y_train)
    # Verificar a qualidade do modelo nos dados de teste ------------------------------------------
    r2_score = modelo.score(X_test, y_test)
    print('R² Score (coeficiente de determinação): ',r2_score)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

ACAO = 'Testar o modelo'
try:
    print(f'\n{ACAO}...')

    # Prever recuperação_veiculos para roubo_veiculo = 400.000, 500.000 e 600.000.
    array_roubo_veiculo_pred = np.array([400000,500000,600000])
    # Normalizar dados que serão utilizados para previsão
    array_roubo_veiculo_pred_scaled = scaler.transform(array_roubo_veiculo_pred.reshape(-1,1))
    # Prever recuperacao_veiculos: método predict
    recup_pred = modelo.predict(array_roubo_veiculo_pred_scaled)
    print('Previsão de recuperação de veículos: ', recup_pred)

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()


#___________________________________________________________________________________________________
#___________________________________VISUALIZANDO OS DADOS __________________________________________

# Avaliação do Modelo: Teste interno ----------------------------------------------------------------

ACAO = 'Avaliar o modelo de previsões'
try:
    print(f'\n{ACAO}...')

    # ______________________________________________________________________________________________
    # ------------------------------------------------------------------------------Título do Painel
    plt.subplots(2,2, figsize=(15,5))
    plt.suptitle('Avaliação do modelo de regressão')

    # Layout do Painel: 2 linhas e 2 colunas
    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 1
    # Dispersão entre os arrays
    plt.subplot(2,2,1)
    sns.regplot(x=array_roubo_veiculo, y=array_recuperacao_veiculos)
    plt.title('Gráfico de Dispersão')
    plt.xlabel('Roubo de veículos')
    plt.ylabel('Recuperação Veículos')
    plt.text(min(array_roubo_veiculo), max(array_recuperacao_veiculos -1000), f'Correlação: {correlacao}', fontsize=10)

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 2
    # Dispersão entre dados reais e dados previstos
    # Base dos dados X (roubo_veiculo)
    plt.subplot(2,2,2)
    # Testar o modelo preditivo nos dados de teste
    y_pred = modelo.predict(X_test)
    # Retornar os dados de teste para a escala real
    X_test = scaler.inverse_transform(X_test)
 
    plt.scatter(X_test, y_test, color='blue', label='Dados reais')
    plt.scatter(X_test, y_pred, color='red', label='Previsões')
    plt.title('Dispersão entre dados reais e dados previstos')
    plt.xlabel('Roubo de Veículos')
    plt.ylabel('Recuperações de Veículos')
    plt.legend()

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 3
    # Resíduos: Diferença entre os dados reais e os dados previstos
    plt.subplot(2,2,3)
    # Cãlculo dos resíduos
    residuos = y_test - y_pred

    plt.scatter(y_pred, residuos) # Gráfico de dispersão
    plt.axhline(y=0, color='black', linewidth=2) # Adicionar linha de constante no 0
    plt.title('Resíduos')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')

    # ______________________________________________________________________________________________
    # -------------------------------------------------------------------------------------Gráfico 4
    # Dispersão dos valores simulados
    plt.subplot(2,2,4)
    plt.scatter(array_roubo_veiculo_pred, recup_pred) 
    plt.title('Recuperações de veículos simuladas')
    plt.xlabel('Roubo de veículos simulado')
    plt.ylabel('Recuperação de veículo prevista')

    # ______________________________________________________________________________________________
    plt.tight_layout()
    plt.show()

    print(f'Comando ({ACAO}) executado com sucesso!')
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()