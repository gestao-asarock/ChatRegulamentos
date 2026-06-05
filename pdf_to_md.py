"""
Converte todos os PDFs de /regulamentos/ em Markdown em /regulamentos_md/.

Instalação: pip install pymupdf4llm
Uso: python 2_pdf_to_md.py
"""

import re
import sys
from pathlib import Path

import pymupdf4llm

INPUT_DIR = Path("regulamentos")
OUTPUT_DIR = Path("regulamentos_md")

# Linha de "imagem omitida" gerada pelo pymupdf4llm
_OMITTED_IMAGE = re.compile(r"^\*\*==> picture .* omitted <==\*\*\s*$")
# Assinatura digital no topo/rodapé de algumas páginas
_ENVELOPE_ID = re.compile(r"^(Docusign|ClickSign)\s+Envelope\s+ID:", re.IGNORECASE)
# Pontinhos de sumário seguidos de número de página: "..... 14"
_DOT_LEADER = re.compile(r"\.{3,}\**\s*\d")
# Romano ≥2 chars colado em palavra (II, III … XV) — I e V sozinhos são início de palavras PT
_ROMAN_FUSED = re.compile(
    r"\b(XV|XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II)([a-záàãâéêíóôõú])"
)


def _is_toc_page(text: str) -> bool:
    """Retorna True se a página for predominantemente sumário (pontinhos + números)."""
    return len(_DOT_LEADER.findall(text)) > 3


def _clean_page(text: str) -> str:
    """Remove ruído pontual de uma página já convertida."""
    lines = []
    for line in text.splitlines():
        if _OMITTED_IMAGE.match(line):
            continue
        if _ENVELOPE_ID.match(line.strip()):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # Problema 2: romano colado no texto ("IIdespesas" → "II despesas")
    text = _ROMAN_FUSED.sub(r"\1 \2", text)
    return text


def _is_orphan_term(para: str) -> bool:
    """
    True quando o parágrafo é só um rótulo de definição sem corpo.
    Detecta: '## **Termo:**', '**Termo:**', '- **Termo:**'
    O último char significativo (ignorando espaços e **) deve ser ':'.
    """
    s = para.strip()
    if not s or len(s) > 200:
        return False
    return bool(re.sub(r"[\s*]+$", "", s).endswith(":"))


def _fix_disconnected_definitions(text: str) -> str:
    """
    Corrige blocos de definições desconexos causados por layout de 2 colunas no PDF.

    Quando o pymupdf4llm lê a coluna de termos primeiro e a coluna de definições
    depois, o resultado é N termos seguidos de N definições. Esta função detecta
    esse padrão (≥2 termos consecutivos sem definição) e reparea na ordem original.
    """
    paras = [p for p in re.split(r"\n\n+", text.strip()) if p.strip()]

    result: list[str] = []
    i = 0
    while i < len(paras):
        if not _is_orphan_term(paras[i]):
            result.append(paras[i])
            i += 1
            continue

        # Coleta sequência de termos órfãos consecutivos
        terms: list[str] = []
        j = i
        while j < len(paras) and _is_orphan_term(paras[j]):
            terms.append(paras[j])
            j += 1

        n = len(terms)
        if n < 2:
            # Termo único isolado — mantém como está (não temos como parear com segurança)
            result.append(paras[i])
            i += 1
            continue

        # Tenta coletar exatamente n definições logo após o bloco de termos
        k = j
        defs: list[str] = []
        while k < len(paras) and len(defs) < n:
            if _is_orphan_term(paras[k]):
                break  # novo bloco de termos — para
            defs.append(paras[k])
            k += 1

        if len(defs) != n:
            # Não encontrou a quantidade certa — não altera
            result.extend(terms)
            i = j
            continue

        # Reparea: **Termo:** definição
        for term, defn in zip(terms, defs):
            term_clean = re.sub(r"^##\s*", "", term.strip())
            # Remove prefixos de markdown da definição (##, - ) que eram artefatos de coluna
            defn_clean = re.sub(r"^(?:##\s+|[-•]\s+)+", "", defn.strip())
            result.append(f"{term_clean} {defn_clean}")
        i = k

    return "\n\n".join(result)


def convert_pdf(pdf_path: Path) -> Path:
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        ignore_images=True,
        ignore_graphics=True,
        page_separators=False,
        page_chunks=True,
        margins=(0, 50, 0, 50),  # ignora 50pt de topo/rodapé (headers e footers do PDF)
    )

    parts: list[str] = []
    has_content = False  # True após encontrar a primeira página de conteúdo real
    for chunk in chunks:
        page_text = chunk["text"]
        if _is_toc_page(page_text):
            continue
        cleaned = _clean_page(page_text)
        if not cleaned.strip():
            continue
        # Pula página de capa (curta, sem heading ##) antes de encontrar conteúdo real
        if not has_content and len(cleaned) < 500 and "##" not in cleaned:
            continue
        has_content = True
        parts.append(cleaned.strip())

    text = "\n\n".join(parts)
    # Problema 1: termos e definições desconexos por layout 2 colunas
    text = _fix_disconnected_definitions(text)

    out_path = OUTPUT_DIR / pdf_path.with_suffix(".md").name
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Pasta de entrada não encontrada: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("**/*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em {INPUT_DIR}.", file=sys.stderr)
        sys.exit(1)

    for pdf in pdfs:
        try:
            out = convert_pdf(pdf)
            print(f"OK  {pdf} -> {out}")
        except Exception as exc:
            print(f"ERRO {pdf}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
