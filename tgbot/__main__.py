import asyncio
import logging

import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import parse_mode
from aiogram.filters import Command
from aiogram.types import BotCommand
from loguru import logger

from .settings import settings

bot = Bot(token=settings.TGBOT_TOKEN)
dp = Dispatcher()


async def set_bot_commands():
    commands = [
        BotCommand(command="start", description="Начать работу с ботом"),
        BotCommand(command="history", description="Показать историю запросов"),
        BotCommand(command="stats", description="Показать статистику запросов"),
    ]
    await bot.set_my_commands(commands)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id  # type: ignore

    if user_id in settings.ALLOWED_USERS:
        await message.answer(
            "Доступ разрешен! Вы можете отправлять текстовые запросы.\n\n"
            "Доступные команды:\n"
            "/history - показать историю ваших запросов\n"
            "/stats - показать статистику"
        )
    else:
        await message.answer("Доступ запрещен. Обратитесь к администратору.")


@dp.message(Command("history"))
async def cmd_history(message: types.Message):
    user_id = message.from_user.id  # type: ignore

    if user_id not in settings.ALLOWED_USERS:
        await message.answer("У вас нет доступа для просмотра истории.")
        return

    processing_msg = await message.answer("Загружаю историю...")
    chat_id = message.chat.id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                settings.API_HISTORY_URL,
                params={"tg_user_id": user_id},
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                try:
                    await bot.delete_message(
                        chat_id=chat_id, message_id=processing_msg.message_id
                    )
                except Exception as e:
                    logging.warning(f"Не удалось удалить сообщение: {e}")

                if response.status == 200:
                    try:
                        history = await response.json()
                        if not history:
                            await message.answer("История запросов пуста.")
                            return

                        text_parts = []
                        for item in history[:10]:
                            status_emoji = (
                                "✅" if item.get("status") == "success" else "❌"
                            )
                            created_at = item.get("created_at", "N/A")
                            request_text = item.get("request_data", {}).get(
                                "text", "N/A"
                            )
                            if len(request_text) > 50:
                                request_text = request_text[:50] + "..."

                            text_parts.append(
                                f"{status_emoji} {created_at[:10]}\n"
                                f"Запрос: {request_text}\n"
                            )

                        result_text = "📜 История ваших запросов:\n\n" + "\n".join(
                            text_parts
                        )
                        if len(history) > 10:
                            result_text += f"\n... и еще {len(history) - 10} запросов"

                        for i in range(0, len(result_text), 4000):
                            await message.answer(result_text[i : i + 4000])
                    except Exception as e:
                        await message.answer(f"Ошибка при обработке истории: {str(e)}")
                else:
                    error_text = await response.text()
                    await message.answer(
                        f"Ошибка API (статус {response.status}):\n{error_text[:500]}"
                    )

    except aiohttp.ClientError as e:
        await message.answer(f"Ошибка соединения с API:\n<code>{str(e)}</code>")
    except asyncio.TimeoutError:
        await message.answer("❌ Таймаут запроса к API")
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка:\n<code>{str(e)}</code>")


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    user_id = message.from_user.id  # type: ignore

    if user_id not in settings.ALLOWED_USERS:
        await message.answer("У вас нет доступа для просмотра статистики.")
        return

    processing_msg = await message.answer("Загружаю статистику...")
    chat_id = message.chat.id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                settings.API_STATS_URL,
                params={"tg_user_id": user_id},
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                try:
                    await bot.delete_message(
                        chat_id=chat_id, message_id=processing_msg.message_id
                    )
                except Exception as e:
                    logging.warning(f"Не удалось удалить сообщение: {e}")

                if response.status == 200:
                    try:
                        stats = await response.json()

                        stats_text = "📊 Статистика ваших запросов:\n\n"
                        stats_text += f"⏱ Среднее время обработки: {stats.get('average_processing_time_ms', 0):.2f} мс\n\n"

                        quantiles = stats.get("quantiles", {})
                        stats_text += "📈 Квантили времени обработки:\n"
                        stats_text += (
                            f"  • Среднее: {quantiles.get('mean', 0):.2f} мс\n"
                        )
                        stats_text += (
                            f"  • Медиана (50%): {quantiles.get('50%', 0):.2f} мс\n"
                        )
                        stats_text += f"  • 95%: {quantiles.get('95%', 0):.2f} мс\n"
                        stats_text += f"  • 99%: {quantiles.get('99%', 0):.2f} мс\n\n"

                        input_chars = stats.get("input_characteristics", {})
                        text_length = input_chars.get("text_length", {})
                        if text_length.get("count", 0) > 0:
                            stats_text += "📝 Характеристики текста:\n"
                            stats_text += (
                                f"  • Всего запросов: {text_length.get('count', 0)}\n"
                            )
                            stats_text += f"  • Средняя длина: {text_length.get('mean', 0):.0f} символов\n"
                            stats_text += f"  • Мин/Макс: {text_length.get('min', 0)}/{text_length.get('max', 0)} символов\n"

                        await message.answer(stats_text)
                    except Exception as e:
                        await message.answer(
                            f"Ошибка при обработке статистики: {str(e)}"
                        )
                elif response.status == 401:
                    await message.answer(
                        "Для просмотра статистики требуется авторизация. Обратитесь к администратору."
                    )
                else:
                    error_text = await response.text()
                    await message.answer(
                        f"Ошибка API (статус {response.status}):\n{error_text[:500]}"
                    )

    except aiohttp.ClientError as e:
        await message.answer(f"Ошибка соединения с API:\n<code>{str(e)}</code>")
    except asyncio.TimeoutError:
        await message.answer("❌ Таймаут запроса к API")
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка:\n<code>{str(e)}</code>")


@dp.message(F.text)
async def handle_text(message: types.Message):
    user_id = message.from_user.id  # type: ignore

    if user_id not in settings.ALLOWED_USERS:
        await message.answer("У вас нет доступа для отправки запросов.")
        return

    user_text = message.text.strip()  # type: ignore

    if user_text.startswith("/"):
        return

    processing_msg = await message.answer("Обрабатываю ваш запрос...")
    processing_msg_id = processing_msg.message_id
    chat_id = message.chat.id

    try:
        await bot.send_chat_action(chat_id, "typing")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.API_URL,
                json={"text": user_text, "tg_user_id": user_id},
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                try:
                    await bot.delete_message(
                        chat_id=chat_id, message_id=processing_msg_id
                    )
                except Exception as e:
                    logging.warning(f"Не удалось удалить сообщение: {e}")

                if response.status == 200:
                    try:
                        data = await response.json()
                        response_text = data["response"]
                    except Exception:
                        response_text = await response.text()

                    escape_chars = r"_*[]()~`>#+-=|{}.!"
                    for char in escape_chars:
                        response_text = response_text.replace(char, f"\\{char}")

                    for i in range(0, len(response_text), 4000):
                        await message.answer(
                            response_text[i : i + 4000],
                            parse_mode=parse_mode.ParseMode.MARKDOWN,
                        )

                else:
                    error_text = await response.text()
                    await message.answer(
                        f"Ошибка API (статус {response.status}):\n{error_text[:1000]}"
                    )

    except aiohttp.ClientError as e:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=processing_msg_id)
        except Exception as ex:
            logger.error(f'msg="Cant delete message" {ex=}')
        await message.answer(f"Ошибка соединения с API:\n<code>{str(e)}</code>")

    except asyncio.TimeoutError:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=processing_msg_id)
        except Exception as ex:
            logger.error(f'msg="Cant delete message" {ex=}')
        await message.answer("❌ Таймаут запроса к API (30 секунд)")

    except Exception as e:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=processing_msg_id)
        except Exception as ex:
            logger.error(f'msg="Cant delete message" {ex=}')
        await message.answer(f"❌ Произошла ошибка:\n<code>{str(e)}</code>")


async def main():
    logging.info("Бот запущен")
    await set_bot_commands()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
