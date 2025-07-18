# app.py (versão aprimorada e corrigida)

import streamlit as st
import google.generativeai as genai
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(
    page_title="MiudinhoAI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiudinhoAI - Análise de Conteúdo")
st.caption("Uma interface para interagir com os vídeos do canal Miudinho Uberaba usando Gemini.")

# --- SEGURANÇA E CONFIGURAÇÃO DA API ---
try:
    # Carregado via st.secrets para segurança
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("ERRO: A chave da API do Gemini não foi encontrada.")
    st.info("Por favor, crie um arquivo .streamlit/secrets.toml e adicione sua chave: GEMINI_API_KEY = 'SUA_CHAVE_AQUI'")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
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
    """Extrai o ID do vídeo da URL e busca a transcrição em português."""
    try:
        video_id = url.split('v=')[-1].split('&')[0]
        if '/' in video_id:
             video_id = video_id.split('/')[-1]

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'pt-BR'])
        transcript_text = " ".join([item['text'] for item in transcript_list])
        return transcript_text
    except (NoTranscriptFound, TranscriptsDisabled):
        st.error("ERRO: Não foi possível encontrar legendas em português para este vídeo.")
        st.warning("A análise não pode prosseguir sem as legendas. Por favor, escolha outro vídeo.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao tentar buscar as legendas: {e}")
        return None

# --- CARREGAMENTO DOS DADOS ---
# ATENÇÃO: Verifique se o caminho do arquivo está correto para o seu ambiente.
json_filepath = r'videos_miudinho_uberaba.json' # Ajustado para um caminho relativo mais simples
video_data = load_video_data(json_filepath)

# --- INTERFACE DO STREAMLIT ---
if video_data:
    video_titles = [video['titulo'] for video in video_data]

    st.header("1. Selecione um Vídeo")
    selected_title = st.selectbox(
        label="Escolha um dos vídeos para analisar:",
        options=video_titles
    )

    selected_video = next((video for video in video_data if video['titulo'] == selected_title), None)

    if selected_video:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"Vídeo Selecionado")
            st.video(selected_video['url'])
        
        with col2:
            st.subheader(f"Informações")
            st.write(f"**Título:** {selected_video['titulo']}")
            if 'descricao' in selected_video and selected_video['descricao']:
                st.write(f"**📜 Versículo-chave:** *{selected_video['descricao']}*")
        
        st.divider()

        st.header("2. Escolha uma Ação")
        
        # *** MELHORIA 2.1: Corrigido o texto da opção para corresponder ao IF abaixo ***
        action = st.radio(
            "O que você gostaria de fazer com este vídeo?",
            ("Análise de Expressões e Referências", "Resumo Inteligente do Vídeo"),
            key="action_choice",
            horizontal=True
        )

        if st.button("Analisar com Gemini", type="primary", use_container_width=True):
            with st.spinner("Buscando legendas do vídeo... 📜"):
                transcript = get_video_transcript(selected_video['url'])

            if transcript:
                # *** MELHORIA 1.1: Uso do modelo 'pro' para maior qualidade de análise ***
                model = genai.GenerativeModel('gemini-2.5-pro')

                # *** MELHORIA 1.2: Configuração de geração para maior consistência ***
                generation_config = genai.types.GenerationConfig(
                    temperature=0.2 
                )
                
                prompt_base = None
                # *** MELHORIA 2.1 (BUG FIX): O texto agora corresponde exatamente ao do st.radio ***
                if action == "Análise de Expressões e Referências":
                    # *** MELHORIA 1.3: Prompt aprimorado para pedir uma saída estruturada em Markdown ***
                    prompt_base = f"""
                    Você é um assistente de pesquisa acadêmica especializado em estudos bíblicos com base na Doutrina Espírita.
                    Sua tarefa é analisar a transcrição de um vídeo e o versículo-chave fornecidos para extrair informações específicas.

                    FORMATE SUA RESPOSTA USANDO MARKDOWN.

                    Com base em AMBOS (a transcrição e o versículo), extraia e liste APENAS o seguinte:
                    
                    ### Palavras e Expressões em Análise
                    Liste a(s) palavra(s) ou expressão(ões) do versículo que são o foco principal da análise no vídeo. Geralmente, o palestrante menciona explicitamente qual termo está "estudando miudinho".

                    ### Referências Bibliográficas
                    Liste todos os livros, autores e capítulos que são explicitamente mencionados no vídeo como fonte de consulta. Use o formato: `Livro (Autor) - Capítulo/Referência`.
                    
                    Se nenhuma referência bibliográfica for mencionada no vídeo, escreva "Nenhuma referência bibliográfica explícita foi mencionada.".
                    Não adicione conclusões, resumos ou qualquer outra informação além do que foi solicitado.
                    """
                
                elif action == "Resumo Inteligente do Vídeo":
                    # *** MELHORIA 1.3: Prompt aprimorado para pedir uma saída estruturada em Markdown ***
                    prompt_base = f"""
                    Você é um especialista em síntese de conteúdo. Sua tarefa é criar um resumo claro e informativo que conecte o conteúdo da transcrição de um vídeo ao seu versículo-chave.

                    FORMATE SUA RESPOSTA USANDO MARKDOWN.

                    Siga estas instruções:
                    
                    ### Resumo da Análise
                    Em 2 a 3 parágrafos, explique como a pregação no vídeo aprofunda e interpreta o tema central apresentado no versículo-chave. O resumo deve ser conciso e fiel ao conteúdo da transcrição.

                    ### Tópicos Principais
                    Liste de 3 a 5 pontos ou argumentos centrais apresentados no vídeo que explicam o versículo.
                    """

                versiculo = selected_video['descricao']
                
                prompt_final = f"""
                {prompt_base}

                --- CONTEXTO PARA ANÁLISE ---

                **VERSÍCULO-CHAVE:**
                {versiculo}

                **TRANSCRIÇÃO COMPLETA DO VÍDEO:**
                {transcript}
                """
                
                with st.spinner("O MiudinhoAI está analisando o conteúdo... 🧠✍️"):
                    try:
                        # *** MELHORIA: Passando a generation_config na chamada ***
                        response = model.generate_content(
                            prompt_final,
                            generation_config=generation_config
                        )
                        st.header("Resultado da Análise")
                        st.markdown(response.text)

                    except Exception as e:
                        st.error(f"Ocorreu um erro ao chamar a API do Gemini: {e}")
                        st.info("Isso pode ocorrer por diversos motivos, como conteúdo bloqueado por políticas de segurança ou um problema temporário na API. Tente novamente mais tarde.")
