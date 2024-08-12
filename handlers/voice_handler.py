from telegram import Update
from models import *
from handlers import start
import tempfile
from telegram.ext import CallbackContext
import subprocess

async def process_voice_message(update: Update, context: CallbackContext):
    user_id = update.effective_user.id  # Get the ID of the user sending the message

    # Update message count
    message_count = await get_message_count(user_id)

    # Check if the user has exceeded the free message limit
    if message_count > 5:
        subscription_status = await get_subscription_status(user_id)
        if subscription_status != 'active':
            await start(update, context)  # Prompt the user to subscribe
            return

    # Download and save the voice message from Telegram
    file = await update.message.voice.get_file()  # Fetch the voice file
    file_bytearray = await file.download_as_bytearray()  # Download the file as a byte array

    # Save the byte array to a temporary OGG file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_ogg:
        temp_ogg.write(file_bytearray)  # Write byte data to the file
        temp_ogg_path = temp_ogg.name  # Get the file path

    # Convert the temporary OGG file to WAV format
    wav_path = temp_ogg_path.replace('.ogg', '.wav')
    subprocess.run(['ffmpeg', '-i', temp_ogg_path, wav_path], check=True)  # Use ffmpeg for conversion

    # Convert the WAV file to text using speech-to-text conversion
    text = speech_to_text_conversion(wav_path)

    # Generate a response based on the text and convert it to speech
    response = await generate_response_gpt(user_id, text, update, context)
    audio_path = await text_to_speech_conversion(response)

    # Send the generated speech response as a voice message
    await send_voice_message(update, context, audio_path)
