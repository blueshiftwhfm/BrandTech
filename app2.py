import streamlit as st
import os
import tempfile
import openai
import random
from streamlit.runtime.scriptrunner import get_script_run_ctx
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, load_prompt
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from time import sleep
from dotenv import load_dotenv
import pyodbc
import json
from datetime import datetime

# # Configurar as informações de conexão
# server = 'srv-futurebrand.database.windows.net'
# database = 'dev-db-futurebrand'
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


# Carregamento das variáveis de ambiente
load_dotenv()

# openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = 'https://blue-chatgpt.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
model = 'blueacademy-embeddings'

llm = AzureChatOpenAI(
    deployment_name="gpt-4-academy",
    model="gpt-4",
    openai_api_version='2023-05-15',
    openai_api_key='b37f67081b8d4a9eae49eebf3476e6fc',
    streaming=True,
    openai_api_base='https://blue-chatgpt.openai.azure.com/',
    temperature=0,
    max_tokens=500
)


embeddings: OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key='b37f67081b8d4a9eae49eebf3476e6fc',
                                                deployment='text-embedding-ada-002',
                                                model_kwargs={'engine': model}, chunk_size=15)

session_id = get_script_run_ctx().session_id
st.experimental_set_query_params(session_id=session_id)


def embedding_and_vector(documents):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200,
                                                   separators=["\\n\\n", "\\n", " ", ""])
    docs = text_splitter.split_documents(documents)
    embeddings_future = embeddings
    db = Chroma.from_documents(docs, embeddings, persist_directory="chromadb/")
    db = Chroma(persist_directory="chromadb/",
                embedding_function=embeddings_future)
    return db, docs


prompt_future = load_prompt('ambiente_virtual\prompt_future.json')
prompt_somos = load_prompt('ambiente_virtual\prompt_somos.json')
prompt_proposito = load_prompt('ambiente_virtual\prompt_proposito.json')
prompt_valor = load_prompt('ambiente_virtual\prompt_valor.json')
prompt_nome_empresa = load_prompt('ambiente_virtual\prompt_nome_empresa.json')

query_nome_empresa = 'Qual é o nome da Empresa'
query_somos = 'Me traga informações sobre a seção de Quem somos/Sobre nós/Sobre da empresa'
query_proposito = 'Me traga informações sobre o Propósito/missão/Visão/Desejo da empresa'
query_valor = 'Me traga informações sobre os valores da empresa'


def respostas_institucionais(qa_somos, qa_proposito, qa_valor, query_somos, query_proposito, query_valor):
    response_somos = qa_somos.run(question=query_somos)
    response_proposito = qa_proposito.run(question=query_proposito)
    response_valor = qa_valor.run(question=query_valor)

    st.write('**Encontrei as seguintes informações institucionais:**')
    st.write('- ' + 'Quem somos: ' + response_somos)
    st.write('- ' + 'Proposito: ' + response_proposito)
    st.write('- ' + 'Valores: ' + response_valor)
    # st.write('- ' + 'Proposta de valor: ' + response_proposta)


########### TESTE WENDER JSON ###########

current_time = datetime.now().strftime("%Y%m%d%H%M")
json_filename = f'chat_history_{current_time}.json'
diretorio = r'C:\Users\BlueShift\OneDrive - blueshift.com.br\Área de Trabalho\VS_Code_-_Blueshift\Future_Brand_Guilherme'
caminho_arquivo = os.path.join(diretorio, json_filename)

if os.path.exists(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as json_file:
        chat_history = json.load(json_file)
    print(f'Arquivo JSON encontrado em {caminho_arquivo}')
else:
    chat_history = []
    print(f'Arquivo JSON não encontrado em {caminho_arquivo}')


def save_messages_to_json(messages, json_filename):
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)

########### TESTE WENDER JSON ###########


def chat(pergunta: str):

    response = qa.run(question=pergunta)

    return response


class SessionMemoryManager:
    def __init__(self):
        self.session_memories = {}

    def get_memory(self, session_id):
        if session_id not in self.session_memories:

            self.session_memories[session_id] = ConversationBufferMemory(
                memory_key="historico",
                return_messages=True,
                human_prefix='Usuário',
                ai_prefix='Bot',
                input_key='pergunta',
                session_id=session_id
            )
        return self.session_memories[session_id]


memory_manager = SessionMemoryManager()

memory = memory_manager.get_memory(session_id)

if 'button' not in st.session_state:
    st.session_state.button = False
    st.info("""         
            No menu a esquerda é possível fazer o upload do arquivo. O processo pode demorar um pouco de
            acordo com o tamanho do documento   
            """)

menu = st.sidebar

st.sidebar.write("Bem-vindo(a) ao Pupila")
st.sidebar.divider()

upload_file = menu.file_uploader("Coloque seu arquivo: ", type=['pdf', 'txt'])

if upload_file is not None:

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá, eu sou o assistente virtual da Pupila! Em que posso ajudar?"}]
    bytes_data = upload_file.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_filename = upload_file.name
        tmp.write(bytes_data)
        tmp.close()
        documents = PyPDFLoader(temp_filename).load()
        db, docs = embedding_and_vector(documents)
        retriever = VectorStoreRetriever(vectorstore=db,
                                         search_kwargs={"filter": {"source": upload_file.name}})

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt_future})

        qa_somos = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt_somos})

        qa_proposito = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt_proposito})

        qa_valor = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt_valor})

        # if st.button('Ver o Quem Somos, Propósito e Valores'):
        #     respostas_institucionais = respostas_institucionais(
        #         qa_somos, qa_proposito, qa_valor, query_somos, query_proposito, query_valor)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Olá, eu sou o assistente virtual da Pupila! Em que posso ajudar?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Digite aqui sua dúvida"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner():  # "Pensando ⌛ ..."):
                    full_response = chat(prompt)
                    full_response_split = full_response.split()

                response_partial = ""

                for palavra in full_response_split:
                    response_partial += palavra + " "
                    message_placeholder.write(response_partial + "▌")
                    tempo = random.uniform(0.02, 0.2)
                    sleep(tempo)

            message_placeholder.write(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})

            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]

            for message in st.session_state.messages:
                chat_history.append(message)
                save_messages_to_json(chat_history, json_filename)

                # with sqlite3.connect(db_filename) as conn:
                #     cursor = conn.cursor()
                #     cursor.execute('''
                #         INSERT INTO messages (role, content)
                #         VALUES (?, ?)
                #     ''', (role, content))

# finally:
#     if conn:
#         conn.close()


# ... (seu código anterior)

# Função para salvar as mensagens em um arquivo JSON

# Carregar histórico de mensagens existentes do arquivo JSON, se houver
# if os.path.exists(json_filename):
#     with open(json_filename, 'r', encoding='utf-8') as json_file:
#         chat_history = json.load(json_file)
# else:
#     chat_history = []
