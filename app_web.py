import os
import re
import json
import streamlit as st
from google import genai
from google.genai import types
import faiss
import pickle
import numpy as np

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(
    page_title="MiudinhoUAI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiudinhoUAI - Busca no Acervo")
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
GENERATIVE_MODEL_NAME = 'gemini-3.5-flash'
EMBEDDING_MODEL_NAME = 'jinaai/jina-embeddings-v3' 

# --- ARQUIVOS E CONSTANTES ---
FAISS_INDEX_FILE = 'banco_vetorial_local_3500.index'
CHUNKS_MAPPING_FILE = 'chunks_mapeamento_local_3500.pkl'
VIDEO_JSON_FILE = 'videos_miudinho_uberaba.json'
SRT_DIR = 'legendas_srt'

# --- IDs DO GOOGLE DRIVE (VOCÊ PRECISA PREENCHER) ---
# Cole aqui apenas o ID dos links de compartilhamento do seu Google Drive
GDRIVE_INDEX_ID = '11fNF1EdMYVR9PuMGB8Skw_M2ffw8ZnhW'
GDRIVE_PKL_ID = '1_f_lP4DjPxQfQQAfDXdgUecE2I8loOXn'

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

@st.cache_data
def load_video_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# Carrega os dados dos vídeos globalmente (se disponível na nuvem/repo)
video_data = load_video_data(VIDEO_JSON_FILE)

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
    # Aplica o prefixo "query:" exigido pelo Jina V3
    queries_com_prefixo = [f"query: {q}" for q in queries]
    
    query_vectors_list = list(model.embed(queries_com_prefixo))
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

def obter_timestamp_do_chunk(source_file_txt, chunk_text):
    if not source_file_txt:
        return 0
    
    # Substitui a extensão .txt por .srt
    source_file_srt = source_file_txt.rsplit('.', 1)[0] + ".srt"
    srt_path = os.path.join(SRT_DIR, source_file_srt)
    
    if not os.path.exists(srt_path):
        return 0
        
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Normaliza quebras de linha e separa por blocos do srt
        content = content.replace('\r\n', '\n')
        srt_blocks = content.strip().split('\n\n')
        
        subtitles = []
        
        # Regex para capturar o tempo inicial do formato: 00:25:20,300 --> 00:25:24,500
        time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->')
        
        for block in srt_blocks:
            lines = block.split('\n')
            if len(lines) < 2:
                continue
            
            # Procura a linha com os tempos
            time_line = lines[1]
            time_match = time_pattern.search(time_line)
            if not time_match:
                # Às vezes a linha de tempo é a primeira se o número do bloco falhar
                time_line = lines[0]
                time_match = time_pattern.search(time_line)
                if not time_match:
                    continue
                    
            start_time_str = time_match.group(1)
            
            # Junta as linhas de texto restantes
            text_lines = []
            for l in lines[2:]:
                if l.strip():
                    text_lines.append(l.strip())
            
            text = " ".join(text_lines)
            
            # Converte start_time_str para segundos
            try:
                time_clean = start_time_str.replace(',', '.')
                parts = time_clean.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                seconds_int = int(hours * 3600 + minutes * 60 + seconds)
                subtitles.append((seconds_int, text))
            except Exception:
                continue
                
        if not subtitles:
            return 0
            
        # Função auxiliar de limpeza para casamento semântico
        def clean_text(t):
            t = t.lower()
            t = re.sub(r'[áàâãä]', 'a', t)
            t = re.sub(r'[éèêë]', 'e', t)
            t = re.sub(r'[íìîï]', 'i', t)
            t = re.sub(r'[óòôõö]', 'o', t)
            t = re.sub(r'[úùûü]', 'u', t)
            t = re.sub(r'[ç]', 'c', t)
            t = re.sub(r'[^a-z0-9]', '', t)
            return t
            
        # Constrói o texto contínuo e o mapa de índices de caracteres para segundos
        full_srt_text = ""
        char_to_seconds = []
        
        for sec, text in subtitles:
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue
            start_idx = len(full_srt_text)
            full_srt_text += cleaned_text
            char_to_seconds.append((start_idx, len(full_srt_text), sec))
            
        if not full_srt_text:
            return 0
            
        # Tenta buscar usando prefixos de tamanho decrescente do chunk para ser tolerante a erros
        search_prefixes = []
        chunk_clean = clean_text(chunk_text)
        
        if len(chunk_clean) >= 60:
            search_prefixes.append(chunk_clean[:60])
        if len(chunk_clean) >= 40:
            search_prefixes.append(chunk_clean[:40])
        if len(chunk_clean) >= 20:
            search_prefixes.append(chunk_clean[:20])
            
        # Fallback: primeira palavra e palavras-chave
        words = chunk_clean.split()
        if len(words) >= 4:
            search_prefixes.append("".join(words[:4]))
            
        for prefix in search_prefixes:
            match_pos = full_srt_text.find(prefix)
            if match_pos != -1:
                # Encontra a qual segundo corresponde o início do casamento
                for start_idx, end_idx, sec in char_to_seconds:
                    if start_idx <= match_pos < end_idx:
                        return sec
                        
        # Se não achar nada, retorna o primeiro segundo do primeiro bloco
        return subtitles[0][0]
        
    except Exception as e:
        print(f"Erro ao buscar timestamp no srt: {e}")
        return 0

def formatar_citacoes_com_links(resposta, chunks_relevantes, video_data):
    if not video_data:
        return resposta
        
    title_to_video = {}
    for v in video_data:
        key = sanitize_filename(v['titulo']) + ".txt"
        title_to_video[key] = v
        
    chunk_links = {}
    for chunk in chunks_relevantes:
        src = chunk['source_file']
        if src in title_to_video:
            video = title_to_video[src]
            segundos = obter_timestamp_do_chunk(src, chunk['text'])
            if segundos > 0:
                chunk_links[src] = f"{video['url']}&t={segundos}s"
            else:
                chunk_links[src] = video['url']
        else:
            chunk_links[src] = None
            
    pattern = re.compile(r'\[([^\]]+\.txt)\]')
    
    def replace_match(match):
        filename = match.group(1)
        if filename in chunk_links and chunk_links[filename]:
            display_name = filename.rsplit('.', 1)[0]
            display_name = display_name.replace(" - Miudinho", "").replace(" {Aluizio Elias}", "")
            return f"[[{display_name}]({chunk_links[filename]})]"
        return match.group(0)
        
    return pattern.sub(replace_match, resposta)

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
            # Limpa buscas antigas para dar feedback visual de carregamento
            if 'resposta_final' in st.session_state:
                del st.session_state['resposta_final']
            if 'chunks_relevantes' in st.session_state:
                del st.session_state['chunks_relevantes']
                
            with st.spinner("Refinando e expandindo a pergunta..."):
                expanded_queries = expand_query_with_gemini(user_query)

            with st.spinner("Buscando trechos relevantes no acervo..."):
                chunks_relevantes = buscar_chunks_relevantes(expanded_queries, index, metadata, k=K_VALUE)
            
            if not chunks_relevantes:
                st.warning("Não foram encontrados trechos relevantes para a sua pergunta.")
            else:
                with st.spinner(f"Sintetizando a resposta com a IA ({GENERATIVE_MODEL_NAME})... 🧠✍️"):
                    resposta_final = gerar_resposta_com_busca(user_query, chunks_relevantes)
                
                # Salva no session_state para persistir re-runs (clique em checkboxes, etc.)
                st.session_state['resposta_final'] = resposta_final
                st.session_state['chunks_relevantes'] = chunks_relevantes
        else:
            st.warning("Por favor, digite uma pergunta.")

    # Renderização persistente dos resultados da busca
    if 'resposta_final' in st.session_state and 'chunks_relevantes' in st.session_state:
        resposta_final = st.session_state['resposta_final']
        chunks_relevantes = st.session_state['chunks_relevantes']
        
        st.subheader("Resposta Gerada")
        
        # Formata as citações do Gemini como links clicáveis dinâmicos com timestamps
        resposta_formatada = formatar_citacoes_com_links(resposta_final, chunks_relevantes, video_data)
        st.markdown(resposta_formatada)

        with st.expander("📚 Ver os trechos exatos recuperados (Fontes)"):
            title_to_video = {sanitize_filename(v['titulo']) + ".txt": v for v in video_data} if video_data else {}
            
            for i, chunk in enumerate(chunks_relevantes):
                st.markdown("---")
                src = chunk['source_file']
                video = title_to_video.get(src)
                
                # Layout premium em duas colunas (Texto na esquerda, Player compacto na direita)
                col_texto, col_video = st.columns([3, 2])
                
                segundos = 0
                if video:
                    segundos = obter_timestamp_do_chunk(src, chunk['text'])
                    
                with col_texto:
                    st.markdown(f"**Trecho {i+1} | Fonte:** `{src}`")
                    if video:
                        if segundos > 0:
                            link_timestamp = f"{video['url']}&t={segundos}s"
                            st.markdown(f"🎥 [**Abrir no YouTube (Nova Aba - {segundos // 60}m {segundos % 60}s)**]({link_timestamp})")
                        else:
                            link_timestamp = video['url']
                            st.markdown(f"🎥 [**Abrir no YouTube (Nova Aba - Início)**]({link_timestamp})")
                    
                    st.info(chunk['text'])
                    
                with col_video:
                    if video:
                        # st.video aceita nativamente o parâmetro start_time em segundos!
                        st.video(video['url'], start_time=int(segundos))
else:
    st.error("⚠️ Banco de vetores não encontrado. Certifique-se de que os arquivos .index e .pkl estão na mesma pasta.")
