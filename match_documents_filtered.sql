-- Variante filtrada de match_documents com suporte a fundo_filter.
-- Executar no SQL Editor do Supabase (Dashboard > SQL Editor).
--
-- Diferenças em relação a match_documents:
--   - Parâmetro fundo_filter text[]: quando informado, restringe a busca aos fundos listados.
--   - Threshold padrão menor (0.65) porque a restrição de fundo já limita o escopo.

create or replace function match_documents_filtered(
    query_embedding      vector(768),
    match_count          int     default 37,
    similarity_threshold float   default 0.65,
    fallback_count       int     default 5,
    fundo_filter         text[]  default null
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
    where
        (fundo_filter is null or rc.fundo = any(fundo_filter))
        and (rc.embedding <=> query_embedding) <= (1 - similarity_threshold)
    order by rc.embedding <=> query_embedding
    limit match_count;

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
        where fundo_filter is null or rc.fundo = any(fundo_filter)
        order by rc.embedding <=> query_embedding
        limit fallback_count;
    end if;
end;
$$;