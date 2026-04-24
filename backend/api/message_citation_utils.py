def remap_response_citations(
    response: str,
    chunks_by_source: dict[int, list[tuple[int, float, str]]],
    sorted_source_references: list,
) -> str:
    """Remap [§N] citations from chunk positions to deduplicated source indexes."""
    import re
    from loguru import logger

    position_to_source_id: dict[int, int] = {}
    for source_id, chunks in chunks_by_source.items():
        for position, _, _ in chunks:
            position_to_source_id[position] = source_id

    source_id_to_index = {
        source.source_id: idx + 1
        for idx, source in enumerate(sorted_source_references)
        if hasattr(source, "source_id")
    }

    logger.debug(f"Citation mapping: {len(position_to_source_id)} positions -> {len(source_id_to_index)} sources")
    logger.debug(f"Position to source: {position_to_source_id}")
    logger.debug(f"Source to index: {source_id_to_index}")

    def remap_single_citation(position: int) -> int | None:
        """Remap a single position to new index, or None if invalid."""
        source_id = position_to_source_id.get(position)
        if source_id is None:
            logger.warning(f"Citation §{position} not found in chunks")
            return None

        new_index = source_id_to_index.get(source_id)
        if new_index is None:
            logger.error(f"Source ID {source_id} (from position {position}) not in final list")
            return None

        return new_index

    def replace_citation_group(match: re.Match[str]) -> str:
        """Replace [§1, §2, §3] or [1, 2, 3] with remapped valid citations."""
        citations_str = match.group(1)
        # Extract all numbers - handles both §1 and plain 1 formats
        positions = re.findall(r'§?(\d+)', citations_str)

        # Remap each position
        valid_indices = []
        for pos_str in positions:
            position = int(pos_str)
            new_idx = remap_single_citation(position)
            if new_idx is not None:
                valid_indices.append(new_idx)

        # Remove duplicates and sort
        valid_indices = sorted(set(valid_indices))

        if not valid_indices:
            # All citations were invalid - remove the whole group
            logger.warning(f"All citations in {match.group(0)} were invalid. Removing.")
            return ""

        # Format as [§1, §2, §3]
        formatted = ", ".join(f"§{idx}" for idx in valid_indices)
        logger.debug(f"Remapped {match.group(0)} -> [{formatted}]")
        return f"[{formatted}]"

    # Match citation groups:
    # - [§1, §2, §3] - with § symbol
    # - [1, 2, 3] - plain numbers
    # - [5] - single number
    # Pattern matches any combination of numbers with optional § and commas/spaces
    result = re.sub(r'\[\s*§?(\d+(?:\s*,?\s*§?\d+)*)\s*\]', replace_citation_group, response)

    # Also handle standalone §N not in brackets (if any)
    def replace_standalone(match: re.Match[str]) -> str:
        position = int(match.group(1))
        new_idx = remap_single_citation(position)
        if new_idx is None:
            return ""
        return f"§{new_idx}"

    result = re.sub(r'§(\d+)(?!\])', replace_standalone, result)

    return result
