import pandas as pd
import json

caminho_do_arquivo_json = r'C:\Users\BlueShift\OneDrive - blueshift.com.br\Área de Trabalho\VS_Code_-_Blueshift\Future_Brand_Guilherme\chat_history_202401072236.json'

with open(caminho_do_arquivo_json, 'r', encoding='utf-8') as arquivo_json:
    dados = json.load(arquivo_json)

# print(dados)

mensagens_assistant = []
mensagens_user = []
# mensagens_user.append('Oi Pupila, vou iniciar a conversa.')

for mensagem in dados:
    if mensagem['role'] == 'assistant':
        mensagens_assistant.append(mensagem['content'])
        mensagens_user.append(None)
    elif mensagem['role'] == 'user':
        mensagens_user[-1] = mensagem['content']

df = pd.DataFrame({'Assistant': mensagens_assistant, 'User': mensagens_user})

print(df)
