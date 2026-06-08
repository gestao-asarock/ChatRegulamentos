import hashlib
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from supabase import create_client

load_dotenv()
# --- Config ---
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
MD_FOLDER       = Path("regulamentos_md")
EMBEDDING_MODEL = "gemini-embedding-001"

_api_keys = [k for k in [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY2"),
    os.environ.get("GEMINI_API_KEY3"),
] if k]
_key_index = 0
_clients   = [genai.Client(api_key=k) for k in _api_keys]

def _client() -> genai.Client:
    return _clients[_key_index]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def split_by_h2(text: str) -> list[dict]:
    parts = re.split(r"^(## .+)$", text, flags=re.MULTILINE)
    chunks = []

    preamble = parts[0].strip()
    if preamble:
        chunks.append({"secao": "__preambulo__", "texto": preamble})

    for i in range(1, len(parts) - 1, 2):
        header  = parts[i].strip().lstrip("# ").strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            chunks.append({"secao": header, "texto": content})

    return chunks


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def stored_hash(fundo: str) -> str | None:
    result = (
        supabase.table("regulamentos_chunks")
        .select("file_hash")
        .eq("fundo", fundo)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["file_hash"]
    return None


def delete_chunks(fundo: str) -> None:
    supabase.table("regulamentos_chunks").delete().eq("fundo", fundo).execute()


_backoff = 4
_MAX_BACKOFF = 64


def embed(text: str) -> list[float]:
    global _key_index, _backoff
    keys_tried = 0

    while True:
        try:
            response = _client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768},
            )
            _backoff = 4  # reseta ao ter sucesso
            return response.embeddings[0].values
        except Exception as e:
            if "429" not in str(e):
                raise

            _key_index = (_key_index + 1) % len(_api_keys)
            keys_tried += 1

            if keys_tried < len(_api_keys):
                print(f"          429 — trocando para chave {_key_index + 1}")
            else:
                if _backoff > _MAX_BACKOFF:
                    raise
                print(f"          todas as chaves falharam — aguardando {_backoff}s")
                time.sleep(_backoff)
                _backoff *= 2
                keys_tried = 0


def index_file(md_path: Path) -> None:
    fundo   = md_path.stem
    current = file_hash(md_path)
    saved   = stored_hash(fundo)

    if saved == current:
        print(f"[SKIP]    {fundo} — sem alterações")
        return

    if saved is not None:
        delete_chunks(fundo)
        print(f"[UPDATE]  {fundo} — regulamento atualizado, reindexando")
    else:
        print(f"[NEW]     {fundo}")

    text   = md_path.read_text(encoding="utf-8")
    chunks = split_by_h2(text)

    if not chunks:
        print(f"          nenhum chunk encontrado, pulando")
        return

    rows = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"          chunk {idx}/{len(chunks)}: {chunk['secao'][:60]}")
        vector = embed(chunk["texto"])
        rows.append({
            "fundo":     fundo,
            "secao":     chunk["secao"],
            "texto":     chunk["texto"],
            "embedding": vector,
            "file_hash": current,
        })

    supabase.table("regulamentos_chunks").insert(rows).execute()
    print(f"          {len(rows)} chunks salvos")


def main() -> None:
    md_files = sorted(MD_FOLDER.glob("*.md"))
    if not md_files:
        print(f"Nenhum arquivo .md encontrado em {MD_FOLDER}/")
        return

    print(f"{len(md_files)} arquivo(s) encontrado(s)\n")
    for md_path in md_files:
        index_file(md_path)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
