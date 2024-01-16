# import pyodbc

# # Configurar as informações de conexão
# server = 'srv-futurebrand.database.windows.net'
# database = 'db-futurebrand'
# username = 'admin-future'
# password = 'a2aee9ac7597440f8b4f83f22bdc4302'
# driver = '{ODBC Driver 17 for SQL Server}'

# # Construir a string de conexão
# conn_str = f'SERVER={server};DATABASE={database};UID={username};PWD={password};DRIVER={driver};'

# # Conectar ao banco de dados
# try:
#     conn = pyodbc.connect(conn_str)
#     cursor = conn.cursor()
#     print("Conexão bem-sucedida!")
# except pyodbc.Error as e:
#     print(f"Erro de conexão ao banco de dados: {str(e)}")

# from datetime import datetime


# current_time = datetime.now().strftime("%Y%m%d%H%M")
# json_filename = f'chat_history_{current_time}.json'
# print(json_filename)

import os
import json
# diretorio = r'C:\Users\BlueShift\OneDrive - blueshift.com.br\Área de Trabalho\VS_Code_-_Blueshift\Future_Brand_Guilherme'

# # Lista todos os arquivos no diretório
# arquivos = os.listdir(diretorio)

# # Exibe os nomes dos arquivos
# for arquivo in arquivos:
#     print(arquivo)

caminho_do_arquivo_json = r'C:\Users\BlueShift\OneDrive - blueshift.com.br\Área de Trabalho\VS_Code_-_Blueshift\Future_Brand_Guilherme\chat_history_202401072136.json'

# Leitura do arquivo JSON
with open(caminho_do_arquivo_json, 'r', encoding='utf-8') as arquivo_json:
    dados = json.load(arquivo_json)
print(dados)
