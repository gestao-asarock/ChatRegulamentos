-- Retorna o chunk mais similar por fundo — usado como fallback quando nenhum chunk
-- passa o threshold na busca filtrada (múltiplos fundos ou todos os fundos).
-- Executar no SQL Editor do Supabase (Dashboard > SQL Editor).

create or replace function match_documents_best_per_fund(
    query_embedding vector(768),
    fundo_filter    text[] default null
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
language sql stable
as $$
    select distinct on (rc.fundo)
        rc.id,
        rc.fundo,
        rc.secao,
        rc.texto,
        rc.file_hash,
        (1 - (rc.embedding <=> query_embedding))::float as similarity,
        true as is_fallback
    from regulamentos_chunks rc
    where fundo_filter is null or rc.fundo = any(fundo_filter)
    order by rc.fundo, rc.embedding <=> query_embedding
$$;