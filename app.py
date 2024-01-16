import streamlit as st
import os
import tempfile
import openai
import random
from streamlit.runtime.scriptrunner import get_script_run_ctx
from langchain.memory.chat_message_histories import CosmosDBChatMessageHistory
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

# Carregamento das variáveis de ambiente
load_dotenv()

# Configurar as informações do COSMOS DB
server = 'srv-futurebrand.database.windows.net'
database = 'dev-db-futurebrand'
username = 'admin-future'
password = 'a2aee9ac7597440f8b4f83f22bdc4302'

# Configuração do CosmosDB
cosmosdb_config = {
    "endpoint": "srv-futurebrand.database.windows.net",
    "key": "a2aee9ac7597440f8b4f83f22bdc4302",
    "database_name": "dev-db-futurebrand",
    "container_name": "futurebrand",
}

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


prompt_future = load_prompt('prompts\prompt_future.json')
prompt_somos = load_prompt('prompts\prompt_somos.json')
prompt_proposito = load_prompt('prompts\prompt_proposito.json')
prompt_valor = load_prompt('prompts\prompt_valor.json')
prompt_proposta = load_prompt('prompts\prompt_proposta.json')

query_somos = 'Me traga informações sobre a seção de Quem somos/Sobre nós/Sobre da empresa'
query_proposito = 'Me traga informações sobre o propósito/missão/visão/desejo da empresa'
query_valor = 'Me traga informações sobre os valores da empresa'
query_proposta = 'Me traga informações sobre a proposta de valor/promessa/posicionamento/Brand Statement da empresa'


def respostas_institucionais(query_somos, query_proposito, query_valor, query_proposta):
    response_somos = qa_somos.run(question=query_somos)
    response_proposito = qa_proposito.run(question=query_proposito)
    response_valor = qa_valor.run(question=query_valor)
    response_proposta = qa_proposta.run(question=query_proposta)

    st.write('**Encontrei as seguintes informações institucionais:**')
    st.write('- ' + 'Quem somos: ' + response_somos)
    st.write('- ' + 'Proposito: ' + response_proposito)
    st.write('- ' + 'Valores: ' + response_valor)
    st.write('- ' + 'Proposta de valor: ' + response_proposta)


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

        qa_proposta = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt_proposta})

        respostas_institucionais(
            query_somos, query_proposito, query_valor, query_proposta)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá, eu sou o assistente virtual da Pupila! Em que posso ajudar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        cosmosdb_history.save_messages(session_id, st.session_state.messages)

    if prompt := st.chat_input("Digite aqui sua dúvida"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Pensando ⌛ ..."):
                full_response = chat(prompt)
                full_response_split = full_response.split()

            response_partial = ""

            for palavra in full_response_split:
                response_partial += palavra + " "
                message_placeholder.write(response_partial + "▌")
                tempo = random.uniform(0.05, 0.2)
                sleep(tempo)

        message_placeholder.write(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
