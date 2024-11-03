# Modelo com análise de dados em larga escala.
# Totalizando pagamentos do bolsa família por estados nos meses de janeiro a maio de 2024.
# Arquivo de cálculo.

#___________________________________________________________________________________________________
#________________________________PREPARANDO O VS CODE_______________________________________________

# Bibliotecas --------------------------------------------------------------------------------------
import polars as pl
# Funções: -----------------------------------------------------------------------------------------
from datetime import datetime # Aferir tempo de execução

#___________________________________________________________________________________________________
#________________________________________OBTENDO DADOS______________________________________________

# FONTE DE DADOS: ----------------------------------------------------------------------------------
# Novo Bolsa Família
# Exercício 2024, meses Janeiro a Maio (arquivo parquet)
# CONSTANTES:
ENDERECO_DADOS = r'./DADOS/'

ACAO = 'Obter dados arquivo parquet'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')

    df_bf = pl.read_parquet(ENDERECO_DADOS + 'dados_df_bf_2024.parquet')
    print(df_bf.head())

    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()

#___________________________________________________________________________________________________
#____________________________________TRABALHANDO OS DADOS __________________________________________

# Totalizando o valor da parecela por UF  ----------------------------------------------------------

ACAO = 'Totalizar o valor da parcela por UF'
try:
    hora_inicio = datetime.now()
    print(f'\n{ACAO}...')

    # Substituir Vírgula por Ponto e transformar os dados da coluna VALOR PARCELA em float
    df_bf = df_bf.with_columns(pl.col('VALOR PARCELA').str.replace(',','.').cast(pl.Float64))
    # Totalizar VALOR PARCELA por UF
    df_bf_uf = df_bf.group_by('UF').agg(pl.col('VALOR PARCELA').sum())
    print(df_bf_uf)
    # Desabilitar notação científica:
    pl.Config.set_float_precision(2) # 2 Casas Decimais
    pl.Config.set_decimal_separator(',') # Separador de casas decimais como vírgula
    pl.Config.set_thousands_separator('.') # Separador de casas de milhar como ponto
    # Exibir o data frame completo
    pl.Config.set_tbl_rows(-1) # -1 exibirá todas as linhas 
    # Ordenar do maior par ao menor
    print(df_bf_uf.sort('VALOR PARCELA', descending=True))

    print(f'Comando ({ACAO}) executado com sucesso!')
    hora_fim = datetime.now()
    print('Duração do processamento: ', hora_fim - hora_inicio)
except Exception as e:
    print(f'Erro ao {ACAO}: ', e)
    exit()