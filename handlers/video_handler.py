import tempfile
import os
import cv2
from telegram import Update
from telegram.ext import ContextTypes
from models import *
from handlers import *

# Function to process video messages and extract information
async def process_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # Update message count
    message_count = await get_message_count(user_id)

    # Check if the user has exceeded the free message limit
    if message_count > 5:
        subscription_status = await get_subscription_status(user_id)
        if subscription_status != 'active':
            await start(update, context)  # Prompt the user to subscribe
            return

    file = await update.message.video.get_file()
    video_bytearray = await file.download_as_bytearray()

    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_file:
        tmp_video_file.write(video_bytearray)
        video_path = tmp_video_file.name

    # Extract audio from the video
    audio_path = extract_audio(video_path)

    # Transcribe the audio
    transcript = speech_to_text_conversion(audio_path)

    # Extract frames and describe them
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    frame_interval = fps
    try:
        descriptions = await extract_frames_from_video(video_path, frame_interval)
    except ValueError as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=str(e))
        os.remove(video_path)
        os.remove(audio_path)
        return

    if not descriptions:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Es konnten keine Frames extrahiert werden.")
        os.remove(video_path)
        os.remove(audio_path)
        return

    # Summarize the descriptions
    summary = await summarize_descriptions(descriptions)

    # Create a response
    response_text = f"""Bitte reagiere auf das, was du im Video siehst und hörst.
    Der Patient hat dir ein Video geschickt. Du siehst folgendes: {summary}.
    Wenn du nur eine Person siehst - gehe immer davon aus dass es der Patient ist.

    Außerdem verstehst du, was der Patient sagt. Du hörst: {transcript}.

    Beschreibe was du siehst und hörst zurück an den Patienten. Und stelle ihm Fragen dazu als Therapeutin. Reagiere als wärst du
    ein echter Mensch der gerade ein Video von seinem Patienten gesendet bekommen hat und auf den Inhalt reagiert. Du musst nicht auf jedes Detail
    reagieren, gehe davon aus dass das Video vom Patienten kommt und er dir seine Umgebung zeigt. Rede nicht von Szenen oder Video sondern
    sage einfach was du siehst und hörst, z.B. 'Hey das sieht aber nach einer gemuetlichen Wohnung aus, tolle Kueche und Couch - was treibst du da gerade
    spielst du etwa ein Videospiel?'"""

    # Generate a response from GPT-3
    gpt3_response = await generate_response_gpt(user_id, response_text, update, context)

    audio_path = await text_to_speech_conversion(gpt3_response)

    # Send the response as a voice message
    await send_voice_message(update, context, audio_path)

    # Clean up temporary files
    os.remove(video_path)
