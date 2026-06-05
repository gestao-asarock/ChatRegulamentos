import os
from google import genai
from google.genai import types
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY2"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

SIMILARITY_THRESHOLD = 0.72   # mínimo de similaridade para incluir um chunk
MAX_CHUNKS = 37                # máximo de chunks acima do threshold
FALLBACK_COUNT = 5             # chunks retornados se nenhum passa o threshold
DEBUG = False                  # True = mostra scores e contagem de chunks


def embed_query(client: genai.Client, text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return response.embeddings[0].values


def search_chunks(supabase, embedding: list[float]) -> list[dict]:
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": MAX_CHUNKS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "fallback_count": FALLBACK_COUNT,
        },
    ).execute()
    return result.data or []


def inspect_similarity(supabase, embedding: list[float]) -> None:
    """Mostra distribuição de similaridade sem gerar resposta — para calibrar threshold."""
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": 100,
            "similarity_threshold": 0.0,
            "fallback_count": 0,
        },
    ).execute()
    chunks = result.data or []
    if not chunks:
        print("Nenhum chunk no corpus.")
        return

    scores = [c["similarity"] for c in chunks]
    breakpoints = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    print(f"\nDistribuição de similaridade ({len(scores)} chunks totais):")
    for threshold in breakpoints:
        count = sum(1 for s in scores if s >= threshold)
        bar = "█" * min(count, 40)
        print(f"  >= {threshold:.2f}  {bar} {count}")

    print(f"\n  Top-5 mais similares:")
    for c in chunks[:5]:
        print(f"    sim={c['similarity']:.4f}  {c['fundo']} — {c['secao'][:55]}")
    print()


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        fundo = chunk.get("fundo", "N/A")
        secao = chunk.get("secao", "N/A")
        texto = chunk.get("texto", "")
        context_blocks.append(f"[{i}] Fundo: {fundo} | Seção: {secao}\n{texto}")

    context = "\n\n".join(context_blocks)

    return f"""Você é um assistente especializado em regulamentos de fundos de investimento brasileiros.
Responda à pergunta abaixo com base EXCLUSIVAMENTE nas passagens de regulamento fornecidas.
Se a informação não estiver disponível no contexto, diga explicitamente que não encontrou essa informação nos regulamentos consultados.
Não invente informações nem use conhecimento externo.

--- CONTEXTO ---
{context}
--- FIM DO CONTEXTO ---

Pergunta: {question}

Resposta:"""


def format_sources(chunks: list[dict]) -> str:
    seen = set()
    lines = []
    for chunk in chunks:
        fundo = chunk.get("fundo", "N/A")
        secao = chunk.get("secao", "N/A")
        sim = chunk.get("similarity", 0.0)
        key = (fundo, secao)
        if key not in seen:
            seen.add(key)
            suffix = f"  [sim={sim:.4f}]" if DEBUG else ""
            lines.append(f"  • {fundo} — {secao}{suffix}")
    return "\n".join(lines)


def main():
    gemini = genai.Client(api_key=GEMINI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("RAG — Regulamentos de Fundos de Investimento")
    print("Comandos especiais: '!inspect <pergunta>' mostra distribuição de similaridade sem gerar resposta.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            raw = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not raw:
            continue
        if raw.lower() == "sair":
            print("Encerrando.")
            break

        # Modo inspeção: mostra distribuição de similaridade sem chamar o LLM
        if raw.startswith("!inspect "):
            question = raw[len("!inspect "):].strip()
            if question:
                print("Calculando distribuição de similaridade...")
                embedding = embed_query(gemini, question)
                inspect_similarity(supabase, embedding)
            continue

        question = raw
        print("Buscando chunks relevantes...")
        embedding = embed_query(gemini, question)
        chunks = search_chunks(supabase, embedding)

        if not chunks:
            print("Nenhum chunk encontrado para esta pergunta.\n")
            continue

        using_fallback = chunks[0].get("is_fallback", False)

        if DEBUG:
            mode = "FALLBACK" if using_fallback else f"threshold={SIMILARITY_THRESHOLD}"
            print(f"[DEBUG] {len(chunks)} chunks recuperados ({mode})")
        elif using_fallback:
            print(f"  Aviso: nenhum chunk com sim>={SIMILARITY_THRESHOLD} — usando {len(chunks)} melhores disponíveis.")

        prompt = build_prompt(question, chunks)

        print("Gerando resposta...\n")
        response = gemini.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )

        print("=" * 60)
        print(response.text.strip())
        print()
        print("Fontes consultadas:")
        print(format_sources(chunks))
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
