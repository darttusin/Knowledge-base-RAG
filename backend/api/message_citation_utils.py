def remap_response_citations(
    response: str,
    chunks_by_source: dict[int, list[tuple[int, float, str]]],
    sorted_source_references: list,
) -> str:
    """Remap [§N] citations from chunk positions to deduplicated source indexes."""
    import re

    position_to_source_id: dict[int, int] = {}
    for source_id, chunks in chunks_by_source.items():
        for position, _, _ in chunks:
            position_to_source_id[position] = source_id

    source_id_to_index = {
        source.source_id: idx + 1
        for idx, source in enumerate(sorted_source_references)
        if hasattr(source, "source_id")
    }

    def replace_match(match: re.Match[str]) -> str:
        position = int(match.group(1))
        source_id = position_to_source_id.get(position)
        if source_id is None:
            return match.group(0)

        new_index = source_id_to_index.get(source_id)
        if new_index is None:
            return match.group(0)

        return f"§{new_index}"

    return re.sub(r"§(\d+)", replace_match, response)
