import tempfile
import os
import cv2
import torch
from fer import FER  # If FER is a class from a library for emotion recognition
from PIL import Image
from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes
from config import config
from models import *
from handlers import *

# Import transformer models and OpenAI API
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    BlipProcessor,
    BlipForConditionalGeneration
)

# Load object detection models
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)


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

    # Initialize the FER detector
    detector = FER()

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

    # Open the last frame for object detection and emotion recognition
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Das letzte Frame konnte nicht gelesen werden.")
        os.remove(video_path)
        os.remove(audio_path)
        return

    # Convert the frame to PIL Image format for object detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    detected_objects = [model.config.id2label[label_id.item()] for label_id in results["labels"]]

    # Perform emotion detection on the last frame
    emotion_data = detector.detect_emotions(frame)
    if emotion_data:
        first_entry = emotion_data[0]
        emotions = first_entry['emotions']
        if emotions:
            emotion = max(emotions, key=emotions.get)
            score = emotions[emotion]
        else:
            emotion = "Unbekannt"
            score = 0
    else:
        emotion = "Unbekannt"
        score = 0

    # Create a response
    response_text = f"""Bitte reagiere auf das, was du im Video siehst und hörst.
    Der Patient hat dir ein Video geschickt. Du siehst folgendes: {summary}.
    Du erkennst folgende Emotion: {emotion} (Score: {score}). Wenn du nur eine Person siehst - gehe immer davon aus dass es der Patient ist.

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
