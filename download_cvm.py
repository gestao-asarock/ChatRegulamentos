#!/usr/bin/env python3
"""
Baixa regulamentos de fundos da CVM filtrados por lista de CNPJs.
Fonte: https://dados.cvm.gov.br/dados/FI/DOC/EVENTUAL/
"""

import base64
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

csv.field_size_limit(10_000_000)

ANOS = [2026, 2025, 2024, 2023, 2022]
TP_DOC_REGULAMENTO = {"REGUL FDO", "ALTER REGUL"}

CVM_URL = "https://dados.cvm.gov.br/dados/FI/DOC/EVENTUAL/DADOS/eventual_fi_{ano}.csv"
CNPJS_FILE = Path("cnpjs.txt")
CONTROLE_FILE = Path("controle.json")
REGULAMENTOS_DIR = Path("regulamentos")
TIMEOUT_API = 30
TIMEOUT_DOWNLOAD = 120
DELAY_BETWEEN_DOWNLOADS = 2   # segundos entre PDFs para não levar rate-limit
MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]   # esperas em segundos a cada tentativa


def normalize_cnpj(raw: str) -> str:
    return re.sub(r"[^\d]", "", raw.strip())


def fmt_cnpj(digits: str) -> str:
    d = normalize_cnpj(digits)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}" if len(d) == 14 else d


def load_cnpjs() -> set[str]:
    return {
        normalize_cnpj(line)
        for line in CNPJS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_controle() -> dict:
    if CONTROLE_FILE.exists():
        return json.loads(CONTROLE_FILE.read_text(encoding="utf-8"))
    return {}


def save_controle(controle: dict) -> None:
    CONTROLE_FILE.write_text(
        json.dumps(controle, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fetch_csv_rows(ano: int) -> list[dict]:
    url = CVM_URL.format(ano=ano)
    try:
        resp = requests.get(url, timeout=TIMEOUT_API)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Aviso: erro ao buscar dados de {ano}: {e}")
        return []
    text = resp.content.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text), delimiter=";"))


class InvalidPdfError(Exception):
    pass


def download_pdf_with_retry(url: str, dest: Path) -> None:
    """Baixa PDF com até MAX_RETRIES tentativas e backoff em caso de erro."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT_DOWNLOAD, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                raise InvalidPdfError(f"servidor retornou HTML em vez de PDF (Content-Type: {content_type!r})")
            raw = b"".join(c for c in resp.iter_content(chunk_size=8192) if c)
            # Alguns endpoints retornam o PDF como string JSON base64: "JVBERi0x..."
            if raw[:1] == b'"':
                raw = base64.b64decode(json.loads(raw))
            if raw[:4] != b"%PDF":
                raise InvalidPdfError(f"arquivo não começa com assinatura PDF: {raw[:16]!r}")
            dest.write_bytes(raw)
            return  # sucesso
        except (requests.RequestException, InvalidPdfError) as e:
            if dest.exists():
                dest.unlink()
            is_last = attempt == MAX_RETRIES - 1
            status = getattr(getattr(e, "response", None), "status_code", None)
            if is_last:
                raise
            wait = RETRY_BACKOFF[attempt]
            print(f"    Tentativa {attempt + 1} falhou ({status or e.__class__.__name__}: {e}), aguardando {wait}s...")
            time.sleep(wait)


def main() -> None:
    if not CNPJS_FILE.exists():
        sys.exit("Erro: cnpjs.txt não encontrado.")

    cnpjs = load_cnpjs()
    if not cnpjs:
        sys.exit("Erro: nenhum CNPJ válido em cnpjs.txt.")

    controle = load_controle()
    REGULAMENTOS_DIR.mkdir(exist_ok=True)

    print(f"Buscando regulamentos para {len(cnpjs)} CNPJ(s) nos anos {ANOS}...")
    print("(O script para de baixar CSVs assim que encontrar todos os fundos)\n")

    latest: dict[str, dict] = {}
    remaining = set(cnpjs)  # CNPJs ainda não encontrados

    for ano in ANOS:
        if not remaining:
            print("  Todos os CNPJs já encontrados — pulando anos restantes.")
            break

        print(f"  Consultando {ano}...", end=" ", flush=True)
        rows = fetch_csv_rows(ano)
        matches = 0
        for row in rows:
            cnpj = normalize_cnpj(row.get("CNPJ_FUNDO_CLASSE") or row.get("CNPJ_FUNDO") or "")
            if cnpj not in cnpjs:
                continue
            if row.get("TP_DOC", "").strip() not in TP_DOC_REGULAMENTO:
                continue
            dt_str = (row.get("DT_RECEB") or row.get("DT_COMPTC") or "").strip()
            dt_key = re.sub(r"[^\d]", "", dt_str)[:8]
            link = row.get("LINK_ARQ", "").strip()
            if not dt_key or not link:
                continue
            if cnpj not in latest or dt_key > latest[cnpj]["_dt_key"]:
                latest[cnpj] = {**row, "_dt_key": dt_key}
                remaining.discard(cnpj)
                matches += 1
        print(f"{matches} ocorrência(s) | {len(remaining)} CNPJ(s) ainda sem regulamento")

    print()
    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, (cnpj, row) in enumerate(latest.items()):
        dt_key = row["_dt_key"]
        nome = row.get("DENOM_SOCIAL", "").strip()[:55]

        if controle.get(cnpj) == dt_key:
            skipped += 1
            continue

        link = row["LINK_ARQ"].strip()
        dest = REGULAMENTOS_DIR / f"{cnpj}_{dt_key}.pdf"

        try:
            print(f"  [{i+1}/{len(latest)}] {fmt_cnpj(cnpj)} — {nome} ({dt_key})...")
            download_pdf_with_retry(link, dest)
            controle[cnpj] = dt_key
            save_controle(controle)
            downloaded += 1
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
        except (requests.RequestException, InvalidPdfError) as e:
            failed.append((fmt_cnpj(cnpj), str(e)))
            if dest.exists():
                dest.unlink()

    for cnpj in cnpjs - set(latest.keys()):
        failed.append((fmt_cnpj(cnpj), "não encontrado na base EVENTUAL da CVM"))

    print("\n=== Resumo ===")
    print(f"Fundos verificados  : {len(cnpjs)}")
    print(f"PDFs baixados       : {downloaded}")
    print(f"Já atualizados      : {skipped}")
    print(f"Falhas / não achou  : {len(failed)}")
    if failed:
        print("\nDetalhes das falhas:")
        for cnpj_fmt, reason in failed:
            print(f"  {cnpj_fmt}: {reason}")


if __name__ == "__main__":
    main()
