# Modelo com análise de dados em larga escala.
# Totalizando pagamentos do bolsa família por estados nos meses de janeiro a maio de 2024.
# Arquivo de pré-processamento.

#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# ANTES DE COMEÇAR A ESCREVER O CÓDIGO

# Bibliotecas --------------------------------------------------------------------------------------
import polars as pl
# Funções: -----------------------------------------------------------------------------------------
from auxiliar.conexoes_trein import obter_dados_pl
from datetime import datetime # Aferir tempo de execução

#___________________________________________________________________________________________________
#________________________________________OBTENDO DADOS______________________________________________

# FONTE DE DADOS: ----------------------------------------------------------------------------------
# Novo Bolsa Família
# Exercício 2024, meses Janeiro a Maio
# CONSTANTES:
ENDERECO_DADOS = r'./DADOS/'

ACAO = 'Obter dados de bolsa família'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')

    # Lista de dados do Bolsa Família
    lista_arquivos = ['202401_NovoBolsaFamilia.csv', '202402_NovoBolsaFamilia.csv', \
                      '202403_NovoBolsaFamilia.csv', '202404_NovoBolsaFamilia.csv', \
                        '202405_NovoBolsaFamilia.csv']
    for arquivo in lista_arquivos:
        print('Obtendo dados do arquivo: ', arquivo)
        df = obter_dados_pl(ENDERECO_DADOS + arquivo,'', 'csv',';')            
        if 'df_bf' in locals(): 
            df_bf = pl.concat([df_bf, df])
        else:
            df_bf = df
        print(f'Dados do arquivo {arquivo} obtidos com sucesso!')
    
    print(f'Comando ({ACAO}) executado com sucesso!')
    
    print(df_bf['MÊS COMPETÊNCIA'].unique())
    print(df_bf.head())
    
    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

# Filtrar Dados por CPFs e NISs não nulos e transformar em Parquet ---------------------------------

ACAO = 'Filtrar dados e transformar em parquet'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')
    # Filtrar
    df_bf = df_bf.filter((pl.col('CPF FAVORECIDO').is_not_null)and(pl.col('NIS FAVORECIDO').is_not_null)and(pl.col('CPF FAVORECIDO') !=''))
    # Transformar em Parquet
    df_bf.write_parquet(ENDERECO_DADOS + 'dados_df_bf_2024.parquet')

    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()