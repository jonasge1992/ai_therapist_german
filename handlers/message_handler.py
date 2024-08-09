from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes
from utils.extractors import extract_urls
from models import *
import logging
from config import config
from handlers import *
from utils import *

# Function to process incoming messages
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    user_id = update.message.from_user.id

    # Update message count
    message_count = await get_message_count(user_id)

    # Check if the user has exceeded the free message limit
    if message_count > 5:
        subscription_status = await get_subscription_status(user_id)
        if subscription_status != 'active':
            await start(update, context)  # Prompt the user to subscribe
            return

    try:
        # Extract URLs from the user message
        urls = extract_urls(user_message)
        summary = None

        if urls:
            logging.info(f"Extracted URLs: {urls}")
            try:
                summaries = [await extract_text_html(url) for url in urls]
                summary = " ".join(summaries)
            except Exception as e:
                logging.error(f"Error processing URLs: {e}", exc_info=True)
                summary = "Error processing URLs."

        # Create the GPT-3 prompt
        gpt3_prompt = f"User message: {user_message}\nExtracted URLs: {', '.join(urls) if urls else 'None'} mit folgendem Inhalt {summary}"
        gpt3_response = await generate_response_gpt(user_id, gpt3_prompt, update, context)
        await update.message.reply_text(gpt3_response)

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        await update.message.reply_text("An unexpected error occurred while processing your message. Please try again.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await reset_conversation(user_id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Dein Konversationsverlauf wurde zurückgesetzt.")
