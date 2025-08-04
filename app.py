# app.py (versão final unificada)

import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(
    page_title="MiudinhoAI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiudinhoAI - Análise e Busca de Conteúdo")
st.caption("Uma interface para buscar em todo o acervo ou analisar vídeos individuais do canal Miudinho Uberaba.")

# --- SEGURANÇA E CONFIGURAÇÃO DA API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("ERRO: A chave da API do Gemini não foi encontrada.")
    st.info("Por favor, crie um arquivo .streamlit/secrets.toml e adicione sua chave: GEMINI_API_KEY = 'SUA_CHAVE_AQUI'")
    st.stop()

# --- MODELOS GEMINI (centralizado) ---
GENERATIVE_MODEL = genai.GenerativeModel('gemini-2.5-flash')
EMBEDDING_MODEL = 'models/embedding-001'

# --- ARQUIVOS E CONSTANTES (centralizado) ---
FAISS_INDEX_FILE = 'banco_vetorial_gemini_txt_1500.index'
CHUNKS_MAPPING_FILE = 'chunks_mapeamento_gemini_txt_1500.pkl'
VIDEO_JSON_FILE = 'videos_miudinho_uberaba.json'

# --- FUNÇÕES AUXILIARES DA ABA DE BUSCA (RAG) ---

# @st.cache_resource é ideal para carregar modelos, conexões ou dados pesados que não mudam.
@st.cache_resource
def load_faiss_index():
    """Carrega o índice FAISS e os metadados do disco."""
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CHUNKS_MAPPING_FILE, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    except FileNotFoundError:
        st.error(f"ERRO: Arquivos de banco de vetores ('{FAISS_INDEX_FILE}' ou '{CHUNKS_MAPPING_FILE}') não encontrados!")
        st.warning("Verifique se os arquivos estão no repositório e se o Git LFS foi usado corretamente.")
        return None, None

def buscar_chunks_relevantes(query, index, metadata, k=20):
    """Busca os k chunks mais relevantes e retorna seus metadados."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )
    query_vector = np.array([result['embedding']])
    distances, indices = index.search(query_vector, k)
    return [metadata[idx] for idx in indices[0]]

def gerar_resposta_com_busca(query, chunks_relevantes):
    """Gera uma resposta com base na busca, incluindo citações."""
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
        response = GENERATIVE_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {e}"

# --- FUNÇÕES AUXILIARES DA ABA DE ANÁLISE DE VÍDEO ---

@st.cache_data
def load_video_data(filepath):
    """Carrega os dados dos vídeos a partir de um arquivo JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ERRO: O arquivo de dados '{filepath}' não foi encontrado.")
        return None
    except json.JSONDecodeError:
        st.error(f"ERRO: O arquivo '{filepath}' não é um JSON válido.")
        return None

def get_video_transcript(url):
    """Extrai a transcrição de um vídeo do YouTube."""
    try:
        video_id = url.split('v=')[-1].split('&')[0]
        if '/' in video_id:
             video_id = video_id.split('/')[-1]

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'pt-BR'])
        return " ".join([item['text'] for item in transcript_list])
    except (NoTranscriptFound, TranscriptsDisabled):
        st.error("ERRO: Não foi possível encontrar legendas em português para este vídeo.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar as legendas: {e}")
        return None

# --- INTERFACE PRINCIPAL COM ABAS ---
tab1, tab2 = st.tabs(["**🔍 Busca em Todo o Acervo**", "**🎬 Análise de Vídeo Individual**"])

# --- ABA 1: BUSCA EM TODO O ACERVO ---
with tab1:
    st.header("Busque por temas em todos os estudos")
    st.info("Faça uma pergunta em linguagem natural e o MiudinhoAI buscará a resposta em todos os vídeos com legendas disponíveis, citando as fontes.")

    # Carrega os dados para a busca
    index, metadata = load_faiss_index()

    if index is not None:
        user_query = st.text_input("Digite sua pergunta:", key="search_query")

        if st.button("Buscar Resposta", type="primary", use_container_width=True):
            if user_query:
                with st.spinner("Buscando trechos relevantes no acervo..."):
                    chunks_relevantes = buscar_chunks_relevantes(user_query, index, metadata)
                
                if not chunks_relevantes:
                    st.warning("Não foram encontrados trechos relevantes para a sua pergunta.")
                else:
                    with st.spinner("O MiudinhoAI está sintetizando a resposta... 🧠✍️"):
                        resposta_final = gerar_resposta_com_busca(user_query, chunks_relevantes)
                    
                    st.subheader("Resposta Gerada")
                    st.markdown(resposta_final)
            else:
                st.warning("Por favor, digite uma pergunta.")

# --- ABA 2: ANÁLISE DE VÍDEO INDIVIDUAL (VERSÃO CORRIGIDA E COMPLETA) ---
with tab2:
    st.header("Analise um vídeo específico")
    st.info("Escolha um vídeo da lista para obter um resumo inteligente ou uma análise de expressões e referências.")

    # Carrega os dados dos vídeos para esta aba
    video_data = load_video_data(VIDEO_JSON_FILE)

    if video_data:
        video_titles = [video['titulo'] for video in video_data]
        selected_title = st.selectbox("Escolha um dos vídeos para analisar:", options=video_titles, key="video_selector")
        
        # Encontra o dicionário completo do vídeo selecionado
        selected_video = next((video for video in video_data if video['titulo'] == selected_title), None)

        if selected_video:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.video(selected_video['url'])
            with col2:
                st.subheader("Informações do Vídeo")
                st.write(f"**Título:** {selected_video['titulo']}")
                # Garante que a descrição (versículo) existe e não está vazia antes de mostrar
                if 'descricao' in selected_video and selected_video['descricao']:
                    st.write(f"**📜 Versículo-chave:** *{selected_video['descricao']}*")
            
            st.divider()

            action = st.radio(
                "O que você gostaria de fazer com este vídeo?",
                ("Análise de Expressões e Referências", "Resumo Inteligente do Vídeo"),
                key="action_choice",
                horizontal=True
            )

            if st.button("Analisar com Gemini", key="analyze_button", use_container_width=True):
                with st.spinner("Buscando legendas do vídeo... 📜"):
                    transcript = get_video_transcript(selected_video['url'])

                # A MÁGICA ACONTECE AQUI:
                # Toda a lógica a seguir só é executada SE a transcrição for obtida com sucesso.
                if transcript:
                    prompt_base = ""
                    generation_config = genai.types.GenerationConfig(
                        temperature=0.2 
                    )

                    # 1. Define o prompt base de acordo com a ação escolhida
                    if action == "Análise de Expressões e Referências":
                        prompt_base = f"""
                        Você é um assistente de pesquisa acadêmica especializado em estudos bíblicos com base na Doutrina Espírita.
                        Sua tarefa é analisar a transcrição de um vídeo e o versículo-chave fornecidos para extrair informações específicas.
                        FORMATE SUA RESPOSTA USANDO MARKDOWN.
                        Com base em AMBOS (a transcrição e o versículo), extraia e liste APENAS o seguinte:
                        
                        ### Palavras e Expressões em Análise
                        Liste a(s) palavra(s) ou expressão(ões) do versículo que são o foco principal da análise no vídeo. Geralmente, o palestrante menciona explicitamente qual termo está "estudando miudinho".

                        ### Referências Bibliográficas
                        Liste todos os livros, autores e capítulos que são explicitamente mencionados no vídeo como fonte de consulta. Use o formato: `Livro (Autor) - Capítulo/Referência`.
                        Se nenhuma referência bibliográfica for mencionada, escreva "Nenhuma referência bibliográfica explícita foi mencionada.".
                        Não adicione conclusões ou qualquer outra informação além do que foi solicitado.
                        """
                    
                    elif action == "Resumo Inteligente do Vídeo":
                        prompt_base = f"""
                        Você é um especialista em síntese de conteúdo. Sua tarefa é criar um resumo claro e informativo que conecte a transcrição de um vídeo ao seu versículo-chave.
                        FORMATE SUA RESPOSTA USANDO MARKDOWN.
                        Siga estas instruções:
                        
                        ### Resumo da Análise
                        Em 2 a 3 parágrafos, explique como a pregação no vídeo aprofunda e interpreta o tema central apresentado no versículo-chave. O resumo deve ser conciso e fiel ao conteúdo.

                        ### Tópicos Principais
                        Liste de 3 a 5 pontos ou argumentos centrais apresentados no vídeo que explicam o versículo.
                        """

                    # 2. Constrói o prompt final com todo o contexto
                    versiculo = selected_video.get('descricao', 'Nenhum versículo fornecido.')
                    
                    prompt_final = f"""
                    {prompt_base}

                    --- CONTEXTO PARA ANÁLISE ---
                    **VERSÍCULO-CHAVE:**
                    {versiculo}

                    **TRANSCRIÇÃO COMPLETA DO VÍDEO:**
                    {transcript}
                    """
                    
                    # 3. Chama a API e mostra o resultado
                    with st.spinner("O MiudinhoAI está analisando o conteúdo... 🧠✍️"):
                        try:
                            response = GENERATIVE_MODEL.generate_content(
                                prompt_final,
                                generation_config=generation_config
                            )
                            st.header("Resultado da Análise")
                            st.markdown(response.text)

                        except Exception as e:
                            st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
                            st.info("Isso pode ocorrer por diversos motivos, como conteúdo bloqueado por políticas de segurança ou um problema temporário na API.")