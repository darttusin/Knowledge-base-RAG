from db import RequestHistory
from schemas.stats import StatsResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def get_stats(
    session: AsyncSession, tg_user_id: int | None = None
) -> StatsResponse:
    stmt = select(RequestHistory).where(
        RequestHistory.status == "success",
        RequestHistory.processing_time_ms.isnot(None),
    )
    if tg_user_id is not None:
        stmt = stmt.where(RequestHistory.tg_user_id == tg_user_id)
    result = await session.execute(stmt)
    records = result.scalars().all()

    if not records:
        return StatsResponse(
            average_processing_time_ms=0.0,
            quantiles={"mean": 0.0, "50%": 0.0, "95%": 0.0, "99%": 0.0},
            input_characteristics={},
        )

    # Обработка времени
    processing_times = [
        float(r.processing_time_ms) for r in records if r.processing_time_ms
    ]
    processing_times.sort()

    n = len(processing_times)
    mean_time = sum(processing_times) / n if n > 0 else 0.0
    p50 = processing_times[int(n * 0.5)] if n > 0 else 0.0
    p95 = processing_times[int(n * 0.95)] if n > 0 else 0.0
    p99 = processing_times[int(n * 0.99)] if n > 0 else 0.0

    text_lengths = [
        r.input_length for r in records if r.input_length and r.request_type == "json"
    ]
    token_counts = [r.input_tokens for r in records if r.input_tokens]
    image_sizes = [
        (r.image_width, r.image_height)
        for r in records
        if r.image_width and r.image_height
    ]

    input_characteristics = {
        "text_length": {
            "mean": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
            "count": len(text_lengths),
        },
        "token_count": {
            "mean": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "count": len(token_counts),
        },
        "image_sizes": {
            "widths": [w for w, h in image_sizes],
            "heights": [h for w, h in image_sizes],
            "count": len(image_sizes),
        }
        if image_sizes
        else {"count": 0},
    }

    return StatsResponse(
        average_processing_time_ms=mean_time,
        quantiles={
            "mean": mean_time,
            "50%": p50,
            "95%": p95,
            "99%": p99,
        },
        input_characteristics=input_characteristics,
    )
