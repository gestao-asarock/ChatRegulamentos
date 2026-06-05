import os
import streamlit as st
from google import genai
from supabase import create_client

# Garante que os.environ tem as credenciais antes de importar query.py,
# tanto localmente (via .env já carregado) quanto no cloud (via st.secrets).
for _key in ["GEMINI_API_KEY2", "SUPABASE_URL", "SUPABASE_KEY"]:
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = st.secrets[_key]

from query import (
    embed_query,
    search_chunks,
    build_prompt,
    format_sources,
    GENERATION_MODEL,
    SIMILARITY_THRESHOLD,
)

# ── Clients inicializados uma única vez por processo ──────────────────────────
@st.cache_resource
def get_clients():
    gemini   = genai.Client(api_key=st.secrets["GEMINI_API_KEY2"])
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    return gemini, supabase


# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regulamentos de Fundos",
    page_icon="📋",
    layout="centered",
)
st.title("📋 Consulta de Regulamentos de Fundos")

# ── Estado da sessão ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # Cada item: {"role", "content", "sources", "using_fallback", "fallback_count"}
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
                st.text(msg["sources"])

# ── Input e processamento ─────────────────────────────────────────────────────
if question := st.chat_input("Digite sua pergunta sobre regulamentos..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        gemini_client, supabase_client = get_clients()

        with st.spinner("Buscando trechos relevantes..."):
            embedding = embed_query(gemini_client, question)
            chunks    = search_chunks(supabase_client, embedding)

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
                prompt = build_prompt(question, chunks)
                try:
                    response = gemini_client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=prompt,
                    )
                    answer = response.text.strip()
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        answer = "O modelo está temporariamente sobrecarregado. Tente novamente em alguns instantes."
                    else:
                        answer = f"Erro ao gerar resposta: {e}"
                    sources = ""
                    st.error(answer)
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "sources": "", "using_fallback": False, "fallback_count": 0,
                    })
                    st.stop()

            sources = format_sources(chunks)

            st.markdown(answer)
            with st.expander("Fontes consultadas"):
                st.text(sources)

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       answer,
        "sources":       sources,
        "using_fallback": using_fallback,
        "fallback_count": fallback_count,
    })
