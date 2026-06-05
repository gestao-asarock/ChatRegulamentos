-- Função RPC para busca por similaridade de cosseno com threshold dinâmico.
-- Executar no SQL Editor do Supabase (Dashboard > SQL Editor).
--
-- Requer a extensão pgvector habilitada:
--   create extension if not exists vector;
--
-- Lógica:
--   1. Retorna todos os chunks com similarity >= similarity_threshold, até match_count.
--   2. Se nenhum passar do threshold, retorna os top fallback_count sem filtro (is_fallback = true).
--
-- Parâmetros:
--   query_embedding      — vetor da pergunta (768 dims, gerado com RETRIEVAL_QUERY)
--   match_count          — máximo de chunks a retornar quando há resultados acima do threshold
--   similarity_threshold — mínimo de similaridade de cosseno aceito (0.0–1.0)
--   fallback_count       — quantos chunks retornar quando nenhum passa o threshold

create or replace function match_documents(
    query_embedding      vector(768),
    match_count          int   default 37,
    similarity_threshold float default 0.75,
    fallback_count       int   default 5
)
returns table (
    id          bigint,
    fundo       text,
    secao       text,
    texto       text,
    file_hash   text,
    similarity  float,
    is_fallback boolean
)
language plpgsql stable
as $$
begin
    -- Tenta recuperar chunks acima do threshold
    return query
    select
        rc.id,
        rc.fundo,
        rc.secao,
        rc.texto,
        rc.file_hash,
        (1 - (rc.embedding <=> query_embedding))::float as similarity,
        false as is_fallback
    from regulamentos_chunks rc
    where (rc.embedding <=> query_embedding) <= (1 - similarity_threshold)
    order by rc.embedding <=> query_embedding
    limit match_count;

    -- Fallback: se nenhum chunk passou o threshold, retorna os top-N sem filtro
    if not found then
        return query
        select
            rc.id,
            rc.fundo,
            rc.secao,
            rc.texto,
            rc.file_hash,
            (1 - (rc.embedding <=> query_embedding))::float as similarity,
            true as is_fallback
        from regulamentos_chunks rc
        order by rc.embedding <=> query_embedding
        limit fallback_count;
    end if;
end;
$$;

-- Índice para acelerar a busca (crie APÓS inserir os dados).
-- A cláusula WHERE usa <=> diretamente, o que permite uso do índice ivfflat/hnsw.
-- Ajuste lists conforme o volume (regra: sqrt(n_rows)).
-- create index on regulamentos_chunks
--     using ivfflat (embedding vector_cosine_ops)
--     with (lists = 100);
