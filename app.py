import os
import time
import threading
import streamlit as st
from google import genai
from supabase import create_client

# Garante que os.environ tem as credenciais antes de importar query.py,
# tanto localmente (via .env já carregado) quanto no cloud (via st.secrets).
_SECRET_KEYS = ["GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3",
                "GEMINI_API_KEY4", "GEMINI_API_KEY5", "SUPABASE_URL", "SUPABASE_KEY"]
for _key in _SECRET_KEYS:
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = st.secrets[_key]

from query import (
    embed_query,
    get_all_fundos,
    search_chunks_filtered,
    build_prompt,
    format_sources_md,
    fundo_label,
    load_fundo_names,
    GENERATION_MODEL,
    SIMILARITY_THRESHOLD,
)

@st.cache_data
def get_fundo_names():
    mnemonicos = load_fundo_names(field="mnemonico")
    nomes      = load_fundo_names(field="nome")
    return mnemonicos, nomes

mnemonicos, nomes = get_fundo_names()

# ── Clients inicializados uma única vez por processo ──────────────────────────
@st.cache_resource
def get_clients():
    # Coleta todas as keys Gemini disponíveis (GEMINI_API_KEY, GEMINI_API_KEY2, ...)
    gemini_keys = [
        st.secrets[k]
        for k in ["GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3",
                  "GEMINI_API_KEY4", "GEMINI_API_KEY5"]
        if k in st.secrets
    ]
    gemini_clients = [genai.Client(api_key=k) for k in gemini_keys]
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    return gemini_clients, supabase

@st.cache_data(ttl=3600)
def load_fundos(_supabase):
    return get_all_fundos(_supabase)

# ── Índice de rotação de keys (nível de processo — compartilhado entre reruns) ─
# Usa lista mutável para contornar a limitação de global em Streamlit
_key_state = {"idx": 0}
_key_lock  = threading.Lock()

def generate_with_rotation(clients: list, model: str, prompt: str) -> str:
    """Chama Gemini com rotação de key em 429 e backoff em 503."""
    n = len(clients)
    backoff = 1
    keys_tried = 0
    total_attempts = 0
    max_attempts = n * 4  # evita loop infinito se todas as keys falharem repetidamente

    while total_attempts < max_attempts:
        total_attempts += 1
        client = clients[_key_state["idx"] % n]
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                with _key_lock:
                    _key_state["idx"] = (_key_state["idx"] + 1) % n
                keys_tried += 1
                if keys_tried < n:
                    continue
                time.sleep(backoff)
                backoff = min(backoff * 2, 32)
                keys_tried = 0
            elif "503" in err or "UNAVAILABLE" in err:
                if backoff > 16:
                    raise
                time.sleep(backoff)
                backoff *= 2
            else:
                raise
    raise RuntimeError(f"Limite de {max_attempts} tentativas atingido — todas as keys falharam.")

# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regulamentos de Fundos",
    page_icon="📋",
    layout="centered",
)
st.title("📋 Consulta de Regulamentos de Fundos")

# ── Sidebar — seleção de fundos ───────────────────────────────────────────────
gemini_clients, supabase_client = get_clients()
todos_fundos = load_fundos(supabase_client)

with st.sidebar:
    st.header("Filtro de fundos")
    usar_todos = st.checkbox("Todos os fundos", value=True)

    _fmt = lambda f: fundo_label(f, nomes)

    if usar_todos:
        st.multiselect(
            "Fundos selecionados",
            options=todos_fundos,
            default=todos_fundos,
            format_func=_fmt,
            disabled=True,
            help="Desmarque 'Todos os fundos' para selecionar individualmente.",
        )
        fundo_filter = None
    else:
        selecionados = st.multiselect(
            "Fundos selecionados",
            options=todos_fundos,
            format_func=_fmt,
            default=[],
            placeholder="Escolha um ou mais fundos...",
        )
        fundo_filter = selecionados if selecionados else None

    if fundo_filter is not None:
        n = len(fundo_filter)
        st.caption(f"{n} fundo{'s' if n > 1 else ''} selecionado{'s' if n > 1 else ''}")

# ── Estado da sessão ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Renderiza histórico da sessão ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("using_fallback"):
            st.warning(
                f"Nenhum chunk atingiu o limiar de similaridade ({SIMILARITY_THRESHOLD}). "
                f"Exibindo os {msg['fallback_count']} melhores disponíveis — "
                "a resposta pode ser menos precisa."
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Fontes consultadas"):
                st.markdown(msg["sources"])

# ── Input e processamento ─────────────────────────────────────────────────────
if question := st.chat_input("Digite sua pergunta sobre regulamentos..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos relevantes..."):
            embedding = embed_query(gemini_clients[0], question)
            chunks    = search_chunks_filtered(supabase_client, embedding, fundo_filter)

        if not chunks:
            answer         = "Nenhum trecho encontrado nos regulamentos indexados para esta pergunta."
            sources        = ""
            using_fallback = False
            fallback_count = 0
            st.markdown(answer)
        else:
            using_fallback = chunks[0].get("is_fallback", False)
            fallback_count = len(chunks) if using_fallback else 0

            if using_fallback:
                st.warning(
                    f"Nenhum chunk atingiu o limiar de similaridade ({SIMILARITY_THRESHOLD}). "
                    f"Exibindo os {fallback_count} melhores disponíveis — "
                    "a resposta pode ser menos precisa."
                )

            with st.spinner("Gerando resposta..."):
                prompt = build_prompt(question, chunks, mnemonicos)
                try:
                    answer = generate_with_rotation(gemini_clients, GENERATION_MODEL, prompt)
                except Exception as e:
                    err = ("O modelo está temporariamente sobrecarregado. Tente novamente em instantes."
                           if ("503" in str(e) or "UNAVAILABLE" in str(e))
                           else f"Erro ao gerar resposta: {e}")
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err,
                        "sources": "", "using_fallback": False, "fallback_count": 0,
                    })
                    st.stop()

            sources = format_sources_md(chunks, nomes)

            st.markdown(answer)
            with st.expander("Fontes consultadas"):
                st.markdown(sources)

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       answer,
        "sources":       sources,
        "using_fallback": using_fallback,
        "fallback_count": fallback_count,
    })
