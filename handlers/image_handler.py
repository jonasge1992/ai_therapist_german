from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes
from models import *
from handlers import start
import io
import tempfile
import cv2
from fer import FER
import torch

# Function to process photo messages
async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # Update message count
    message_count = await get_message_count(user_id)

    # Check if the user has exceeded the free message limit
    if message_count > 5:
        subscription_status = await get_subscription_status(user_id)
        if subscription_status != 'active':
            await start(update, context)  # Prompt the user to subscribe
            return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_content = await file.download_as_bytearray()

    # Open and process the image
    image = Image.open(io.BytesIO(file_content)).convert("RGB")
    description = await describe_image(image)

    # Generate a response message
    user_message = (
        f"Der Patient hat dir gerade ein Bild geschickt, tue so als könntest du es wirklich sehen - reagiere NUR auf den Teil ab jetzt:\n"
        f"Ich sehe das Bild vor mir und erkenne folgendes: {description}.\n"
        f"Was fällt dir zu diesen Details auf? Gibt es bestimmte Aspekte des Bildes, die für dich besonders bedeutsam sind?\n"
        f"Ich bin gespannt auf deine Gedanken und Eindrücke zu dem, was du mir geschickt hast.\n"
        f"Verwende keinen Konjuktiv!\n"
        f"Falls du nur eine Person siehst, gehe stehts davon aus dass es der Patient ist."
    )

    gpt3_response = await generate_response_gpt(user_id, user_message, update, context)
    await update.message.reply_text(gpt3_response)
