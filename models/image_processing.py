from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes
from models import *
import io
import tempfile
import cv2
from fer import FER
import torch
# Import transformer models and OpenAI API
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    BlipProcessor,
    BlipForConditionalGeneration
)
from openai import AsyncOpenAI
from config import config

# Load object detection models
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Load image captioning models
processor_describe = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_describe = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])

# Function to describe an image using the image captioning model
async def describe_image(image: Image.Image) -> str:
    inputs = processor_describe(images=image, return_tensors="pt")
    out = model_describe.generate(**inputs)
    description = processor_describe.decode(out[0], skip_special_tokens=True)
    return description




# Function to extract frames from a video for analysis
async def extract_frames_from_video(video_path: str, frame_interval: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Das Video konnte nicht geöffnet werden.")

    descriptions = []
    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            inputs = processor_describe(images=image, return_tensors="pt")
            outputs = model_describe.generate(**inputs)
            description = processor_describe.decode(outputs[0], skip_special_tokens=True)
            descriptions.append(description)

    cap.release()
    return descriptions

# Function to summarize descriptions using GPT-3
async def summarize_descriptions(descriptions: list) -> str:
    combined_descriptions = "\n".join(descriptions)
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ein Video wurde in seine einzelteile (als Bilder) aufgeteilt und jedes Bild erhaelt eine Zusammenfassung. Vereine alle Beschreibungen in ein gesamtheitliches Bild. Entferne Wiederholungen und Redundanzen - da es ein Video war. Du bist die Augen (Sinne) eines anderen Agenten. Der folgende Text ist eine Beschreibung von dem was du siehst, sag was du siehst und fass alles zusammen."},
            {"role": "user", "content": combined_descriptions}
        ],
    )
    summary_response = completion.choices[0].message.content
    return summary_response


# Function to process video messages and extract information
async def process_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # Check subscription status
    subscription_status = await get_subscription_status(user_id)
    if subscription_status != 'active':
        await start_with_payment(update, context)
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
    transcript = recognize_speech_from_path(audio_path)

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
    gpt3_response = await generate_response_gpt(user_id, response_text)

    # Send the response as a voice message
    await send_voice_message(update, context, gpt3_response)

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_path)
