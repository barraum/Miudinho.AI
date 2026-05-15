import os
import streamlit as st
from google import genai
from google.genai import types
import faiss
import pickle
import numpy as np

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(
    page_title="MiudinhoAI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiudinhoAI - Busca no Acervo")
st.caption("Faça perguntas e encontre respostas baseadas no acervo teológico e estudos.")

# --- SEGURANÇA E CONFIGURAÇÃO DAS APIs ---
try:
    # No Streamlit Cloud usaremos st.secrets
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        st.error("ERRO: A chave da API do Gemini não foi encontrada no ambiente ou secrets.")
        st.stop()
except Exception as e:
    st.error(f"Erro ao configurar API Gemini: {e}")
    st.stop()

# --- MODELOS ---
GENERATIVE_MODEL_NAME = 'gemini-2.5-flash'
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' 

# --- ARQUIVOS E CONSTANTES ---
FAISS_INDEX_FILE = 'banco_vetorial_local_1200.index'
CHUNKS_MAPPING_FILE = 'chunks_mapeamento_local_1200.pkl'

# --- IDs DO GOOGLE DRIVE (VOCÊ PRECISA PREENCHER) ---
# Cole aqui apenas o ID dos links de compartilhamento do seu Google Drive
GDRIVE_INDEX_ID = '1mUHavIKQpN4bKoe_OyfEznX9nAPnbWys'
GDRIVE_PKL_ID = '1rHgGIxs4HR4lIXz6nlBnHdz5Yuvw4SAf'

@st.cache_resource
def download_banco_nuvem():
    import gdown
    # Verifica se os arquivos já existem. Se não, baixa do Google Drive.
    if not os.path.exists(FAISS_INDEX_FILE):
        with st.spinner("Baixando o banco de vetores do Google Drive (Isso só acontece uma vez)..."):
            try:
                gdown.download(id=GDRIVE_INDEX_ID, output=FAISS_INDEX_FILE, quiet=False)
            except Exception as e:
                st.error("Erro ao baixar o arquivo .index. Verifique se o ID está correto e o link é público.")
                
    if not os.path.exists(CHUNKS_MAPPING_FILE):
        with st.spinner("Baixando os metadados do Google Drive..."):
            try:
                gdown.download(id=GDRIVE_PKL_ID, output=CHUNKS_MAPPING_FILE, quiet=False)
            except Exception as e:
                st.error("Erro ao baixar o arquivo .pkl. Verifique se o ID está correto e o link é público.")

# Baixa os arquivos caso eles não estejam no servidor
download_banco_nuvem()

# --- FUNÇÕES DE CARREGAMENTO (CACHE) ---

@st.cache_resource
def load_embedding_model():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(CHUNKS_MAPPING_FILE):
        return None, None
    
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CHUNKS_MAPPING_FILE, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Erro ao carregar banco de vetores: {e}")
        return None, None

# --- FUNÇÕES DA ABA DE BUSCA (RAG) ---

def buscar_chunks_relevantes(queries: list, index, metadata, k=10):
    model = load_embedding_model()
    query_vectors_list = list(model.embed(queries))
    query_vectors = np.vstack(query_vectors_list).astype(np.float32)
    
    distances, indices = index.search(query_vectors, k)
    
    unique_indices = set()
    for indice_list in indices:
        for idx in indice_list:
            if idx != -1:
                unique_indices.add(idx)

    return [metadata[idx] for idx in unique_indices]

def gerar_resposta_com_busca(query, chunks_relevantes):
    contexto_formatado = "\n\n--- DOCUMENTOS RELEVANTES PARA CONSULTA ---\n"
    for chunk in chunks_relevantes:
        nome_arquivo_fonte = chunk['source_file']
        contexto_formatado += f"\nDOCUMENTO: {nome_arquivo_fonte}\n"
        contexto_formatado += f"CONTEÚDO:\n'''{chunk['text']}'''\n"
    contexto_formatado += "\n--- FIM DOS DOCUMENTOS RELEVANTES ---\n"

    prompt = f"""
    Você é um assistente teológico especialista. Sua tarefa é responder à pergunta do usuário de forma detalhada e estruturada, utilizando EXCLUSIVAMENTE os trechos de texto fornecidos.

    **Instruções Cruciais:**
    1.  Sintetize uma resposta completa e coesa.
    2.  Ao final de CADA frase que utilize informação de um documento, você DEVE citar o nome do arquivo correspondente usando o formato `[nome do arquivo.txt]`.
    3.  Se o conteúdo não for suficiente, diga "Com base nos trechos fornecidos, não tenho informação suficiente para responder a essa pergunta.".
    4.  Não crie uma seção de "Referências" no final. A citação deve estar no corpo do texto.

    **PERGUNTA DO USUÁRIO:**
    "{query}"

    **DOCUMENTOS PARA CONSULTA:**
    {contexto_formatado}

    Agora, construa sua resposta seguindo todas as instruções.
    """
    try:
        response = gemini_client.models.generate_content(
            model=GENERATIVE_MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {e}"

def expand_query_with_gemini(user_query):
    try:
        prompt = f"""
        Você é um assistente de busca especialista em teologia e estudos bíblicos.
        Gere 4 variações da pergunta do usuário para melhorar a busca em uma base de dados de transcrições de vídeos.
        Concentre-se em sinônimos, conceitos relacionados e formas alternativas de expressar o mesmo significado.
        
        Pergunta Original: "{user_query}"

        Retorne APENAS as perguntas geradas. Liste cada pergunta em uma nova linha. NÃO use marcadores, números ou qualquer outra formatação.
        """
        response = gemini_client.models.generate_content(
            model=GENERATIVE_MODEL_NAME,
            contents=prompt
        )
        expanded_queries = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        expanded_queries.insert(0, user_query)
        return expanded_queries
    except Exception:
        return [user_query]    

# --- INTERFACE PRINCIPAL ---

index, metadata = load_faiss_index()

if index is not None:
    user_query = st.text_input("Digite sua pergunta teológica ou tema de estudo:", key="search_query")
    K_VALUE = 15 

    if st.button("Buscar Resposta", type="primary", use_container_width=True):
        if user_query:
            with st.spinner("Refinando e expandindo a pergunta..."):
                expanded_queries = expand_query_with_gemini(user_query)

            with st.spinner("Buscando trechos relevantes no acervo..."):
                chunks_relevantes = buscar_chunks_relevantes(expanded_queries, index, metadata, k=K_VALUE)
            
            if not chunks_relevantes:
                st.warning("Não foram encontrados trechos relevantes para a sua pergunta.")
            else:
                with st.spinner(f"Sintetizando a resposta com a IA ({GENERATIVE_MODEL_NAME})... 🧠✍️"):
                    resposta_final = gerar_resposta_com_busca(user_query, chunks_relevantes)
                
                st.subheader("Resposta Gerada")
                st.markdown(resposta_final)

                with st.expander("📚 Ver os trechos exatos recuperados (Fontes)"):
                    for i, chunk in enumerate(chunks_relevantes):
                        st.markdown("---")
                        st.markdown(f"**Trecho {i+1} | Fonte:** `{chunk['source_file']}`")
                        st.info(chunk['text'])
        else:
            st.warning("Por favor, digite uma pergunta.")
else:
    st.error("⚠️ Banco de vetores não encontrado. Certifique-se de que os arquivos .index e .pkl estão na mesma pasta.")
