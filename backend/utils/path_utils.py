"""Path manipulation utilities."""

from settings import settings


def strip_source_prefix(source_path: str) -> str:
    """Remove common prefixes from source paths.

    Args:
        source_path: Original source path

    Returns:
        Path with prefix removed if found, otherwise original path
    """
    for prefix in settings.RAG_SOURCE_PATH_PREFIXES:
        if source_path.startswith(prefix):
            return source_path[len(prefix):]
    return source_path
