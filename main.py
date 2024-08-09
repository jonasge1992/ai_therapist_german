from telegram import Update
from telegram.ext import ApplicationBuilder
from config import config
from handlers import start, echo, process_video_message, process_voice_message, process_photo, reset
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, PreCheckoutQueryHandler
)
from models import *


if __name__ == '__main__':
    telegram_token=config["TELEGRAM_TOKEN"]
    application = ApplicationBuilder().token(telegram_token).build()

    # Add handlers for various commands and messages
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
    application.add_handler(MessageHandler(filters.VOICE, process_voice_message))
    application.add_handler(MessageHandler(filters.VIDEO, process_video_message))
    application.add_handler(CommandHandler('reset', reset))
    application.add_handler(MessageHandler(filters.PHOTO, process_photo))

    # Start the bot
    application.run_polling()
