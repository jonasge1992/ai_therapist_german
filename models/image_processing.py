from PIL import Image
from telegram import Update
from telegram.ext import ContextTypes
from models import *
import tempfile
import cv2
# Import transformer models and OpenAI API

from openai import AsyncOpenAI
from config import config
import base64

client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])

open_ai_key = config["OPENAI_API_KEY"]

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

async def describe_image(image: Image.Image) -> str:

    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file, format="JPEG")
        image_path = tmp_file.name

    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this image?"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail":"low"
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )


    return str(response.choices[0])


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
            description = await describe_image(image)
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
    frame_interval = 1.5*fps
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
    gpt3_response = await generate_response_gpt(user_id, response_text)

    # Send the response as a voice message
    await send_voice_message(update, context, gpt3_response)

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_path)
