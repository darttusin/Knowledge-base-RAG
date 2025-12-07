import asyncio
import logging

import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import parse_mode
from aiogram.filters import Command
from loguru import logger
from settings import settings

bot = Bot(token=settings.TGBOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id  # type: ignore

    if user_id in settings.ALLOWED_USERS:
        await message.answer("Доступ разрешен! Вы можете отправлять текстовые запросы.")
    else:
        await message.answer("Доступ запрещен. Обратитесь к администратору.")


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
            async with session.get(
                settings.API_URL,
                params={"text": user_text},
                headers={"accept": "application/json"},
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
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
