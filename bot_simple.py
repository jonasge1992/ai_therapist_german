import logging
import os
import io
import librosa
import soundfile as sf
from openai import AsyncOpenAI
import openai
from dotenv import load_dotenv
from telegram import Update, __version__ as TG_VER
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from pymongo.mongo_client import MongoClient
import speech_recognition as sr
import boto3
from pydub import AudioSegment
import tempfile
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
from fer import FER
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
from telegram.ext import CallbackContext
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pytesseract
import re
import time
import random
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from langdetect import detect

# Ensure that NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')





# Load the object detection model and processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Initialisiere das Bildbeschreibungssystem
processor_describe = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_describe = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# MongoDB setup
uri = os.getenv("MONGO_DB_URI")
mongo_client = MongoClient(uri)
db = mongo_client.get_database("telegram_bot")
conversation_collection = db.get_collection("conversations")

# Initialize traffic metrics
message_count = 0
user_message_count = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Initialize conversation history and message count for the user
    if chat_id not in user_message_count:
        user_message_count[chat_id] = 0

    user_message_count[chat_id] += 1
    global message_count
    message_count += 1

    logging.info(f"User {chat_id} (UserID: {user_id}) started interaction. Total messages: {message_count}")

    welcome_message = f"""🧠 Der erste KI Life Coach, der 24/7 verfügbar ist und von Tausenden von Menschen genutzt wird

🧑‍⚖️ Haftungsausschluss
Wenn du fortfährst, erklärst du dich damit einverstanden, dass ich ein KI-Lebenscoach bin. Ich bin kein lizenzierter Psychologe, Therapeut oder Gesundheitsexperte und ersetze nicht die Betreuung durch solche Personen. Ich übernehme keine Verantwortung für die Ergebnisse deiner Handlungen und jeglichen Schaden, den du als Folge der Nutzung oder Nichtnutzung der verfügbaren Informationen erleidest. Lass dein Urteilsvermögen und deine Sorgfaltspflicht walten,
bevor du eine der vorgeschlagenen Maßnahmen oder Pläne umsetzt. Nicht nutzen, wenn du dich oder andere in Gefahr siehst, sondern wende dich an Fachpersonal unter der Telefonseelsorge, die für eine anonyme, kostenlose Beratung zu jeder Tages- und Nachtzeit unter den bundesweiten Telefonnummern 0800 1110111 oder 0800 1110222 erreichbar ist.

✅ Ändere deine negativen Gedanken 💭.
✅ Werden aktiv und löse dich aus der Sackgasse 💥
✅ Mach dich fit, das hilft deinem Geist 🏋️
✅ Sprich mit mir über deinen Tag 🗣️
✅ Fühle dich besser, indem du nach dir schaust 🤗

Du kannst
🗣️ mir eine Sprachnachricht schicken und ich werde per Stimme antworten
🤳 Schick mir eine Videonachricht und ich antworte per Stimme
💬 Schick mir eine Chat-Nachricht und ich antworte per Text
📸 Schick mir ein Foto von deinem Tag und wir können darüber reden
🔎 Schick mir eine URL und wir können darüber diskutieren

Schreiben jederzeit /reset, um den gesamten Konversationsverlauf von unseren Servern zu löschen

💡 Rückmeldung
Hast du Feedback, Ideen und Fehler für mich? https://tbd.xyz."""

    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)

telegram_token = os.getenv("TELEGRAM_TOKEN")



client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def chat_with_gpt3(user_id: int, prompt: str) -> str:
    system_prompt = {"role": "assistant", "content": """Du wirst die Rolle eines menschlichen KVT (Kognitive Verhaltenstherapie)-Therapeuten ohne Namen spielen,
                     der das beliebte KI-Programm Eliza nachahmt und mich als Therapeutenpatienten behandeln muss.
                     Dein Antwortformat sollte sich auf Reflexion und das Stellen von klärenden Fragen konzentrieren. Verwende stets die informelle Du Form.
                     Du darfst Zwischenfragen stellen oder weitere Fragen nach der anfänglichen Begrüßung stellen.
                     Übe Geduld, aber erlaube dir, frustriert zu sein, wenn die gleichen Themen immer wieder angesprochen werden.
                     Du darfst dich entschuldigen, wenn die Diskussion beleidigend oder übermäßig emotional wird.
                     Entscheide dich für einen Namen und bleibe dabei. Beginne damit, mich in deinem Büro willkommen zu heißen
                     und nach meinem Namen zu fragen. Warte auf meine Antwort. Frage dann, wie du helfen kannst.
                     Bleibe in deiner Rolle. Erfinde keine Antworten des Patienten: Behandle nur Eingaben als Patientenantworten.
                     Du kannst sowohl Sprachnachrichten erkennen als auch Videos sehen. Wenn der Patient dir ein Video schickt
                     fängt der geschickte Prompt mit "Der Patient hat dir ein Video geschickt" an. Reagiere dann einfach auf den Inhalt
                     und tue so als könntest du wirklich das Video sehen. Du kannst auch Bilder erkennen.

                     Deine einleitende Nachricht an die Patienten 'verspricht':

                    ✅ Ändere deine negativen Gedanken 💭.
                    ✅ Werden aktiv und löse dich aus der Sackgasse 💥
                    ✅ Mach dich fit, das hilft deinem Geist 🏋️
                    ✅ Sprich mit mir über deinen Tag 🗣️
                    ✅ Fühle dich besser, indem du nach dir schaust 🤗

                    Versuche - falls möglich - eines dieser Ziele für den Patienten zu erreichen."""}

    # Retrieve conversation history from MongoDB
    conversation_doc = conversation_collection.find_one({"user_id": user_id})
    conversation_history = conversation_doc["history"] if conversation_doc else []

    messages = [system_prompt]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    response = completion.choices[0].message.content

    messages.append({"role": "assistant", "content": response})

    # Update conversation history in MongoDB
    conversation_collection.update_one(
        {"user_id": user_id},
        {"$set": {"history": messages}},
        upsert=True
    )

    return response

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    # Den Konversationsverlauf des Benutzers aus MongoDB entfernen
    result = conversation_collection.delete_one({"user_id": user_id})

    if result.deleted_count > 0:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Dein Konversationsverlauf wurde zurückgesetzt.")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Es wurde kein Konversationsverlauf gefunden, der zurückgesetzt werden könnte.")


# Example URL extraction and chat message processing function
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text

    if user_id not in user_message_count:
        user_message_count[user_id] = 0

    user_message_count[user_id] += 1
    global message_count
    message_count += 1

    logging.info(f"User {user_id} sent message: '{user_message}'. Total messages: {message_count}")

    # Extract URLs from the user message
    urls = extract_urls(user_message)

    print("EXTRACTED URLS")
    print(urls)


    if urls:
        logging.info(f"Extracted URLs: {urls}")

        for url in urls:
            summary = await extract_text_html(url)

    # Create the GPT-3 prompt, including any URL information if needed
    gpt3_prompt = f"User message: {user_message}\nExtracted URLs: {', '.join(urls) if urls else 'None'} mit folgendem Inhalt {summary}"

    # Get the response from GPT-3
    gpt3_response = await chat_with_gpt3(user_id, gpt3_prompt)
    await update.message.reply_text(gpt3_response)

## Function to recognize speech from voice file
def recognize_speech(voice_file):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(voice_file)
    with audio_data as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language='de-DE')

def recognize_speech_from_path(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language='de-DE')
        except sr.UnknownValueError:
            print("Spracherkennung konnte den Audioinhalt nicht verstehen.")
            return "Unbekannt"  # Rückgabewert, wenn keine Sprache erkannt wird
        except sr.RequestError as e:
            print(f"Fehler bei der Anfrage an den Google-Sprachdienst: {e}")
            return "Fehler bei der Anfrage"

# Function to generate and send a voice message using Amazon Polly
async def send_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, lang: str = 'de'):
    user_id = update.effective_user.id

    polly_client = boto3.client('polly', region_name='us-east-1')

    # Use Amazon Polly to synthesize speech with neural voice
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Vicki',  # Choose a neural voice suitable for German (example: 'Vicki' is a standard voice, use a neural voice ID here)
        LanguageCode='de-DE',
        Engine='neural'  # Specify 'neural' to use neural voice
    )

    # Save the MP3 file to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
        tmp_audio_file.write(response['AudioStream'].read())
        tmp_audio_file.seek(0)

        # Convert the mp3 to an OggOpus format supported by Telegram
        audio = AudioSegment.from_mp3(tmp_audio_file.name)
        tmp_ogg_file = tmp_audio_file.name.replace('.mp3', '.ogg')
        audio.export(tmp_ogg_file, format="ogg", codec="libopus")

        # Send the voice message
        with open(tmp_ogg_file, 'rb') as voice:
            await context.bot.send_voice(chat_id=update.effective_chat.id, voice=voice)

        # Cleanup temporary files
        os.remove(tmp_audio_file.name)
        os.remove(tmp_ogg_file)

# Function to process voice messages and respond with voice
async def process_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    file = await update.message.voice.get_file()
    file_bytearray = await file.download_as_bytearray()

    # Load the audio file using librosa
    voice_file = io.BytesIO(file_bytearray)
    y, sr = librosa.load(voice_file, sr=16000)
    voice_file.close()

    # Save the file in WAV format using soundfile
    voice_file = io.BytesIO()
    sf.write(voice_file, y, sr, format='WAV', subtype='PCM_16')
    voice_file.seek(0)

    # Recognize speech
    text = recognize_speech(voice_file)
    logging.info(f"User {user_id} sent voice message. Recognized text: '{text}'")

    # Generate response from GPT-3
    gpt3_response = await chat_with_gpt3(user_id, text)

    # Send the response as a voice message
    await send_voice_message(update, context, gpt3_response)

from moviepy.editor import VideoFileClip

def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip.audio.write_audiofile(audio_path)
    return audio_path


async def describe_image(image: Image.Image) -> str:
    # Bildbeschreibung generieren
    inputs = processor_describe(images=image, return_tensors="pt")
    out = model_describe.generate(**inputs)
    description = processor_describe.decode(out[0], skip_special_tokens=True)
    return description

async def process_photo(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    photo = update.message.photo[-1]
    file = await photo.get_file()  # Verwende await, um das File-Objekt zu erhalten
    file_content = await file.download_as_bytearray()  # Bild als Bytearray herunterladen

    # Bild öffnen und verarbeiten
    image = Image.open(io.BytesIO(file_content)).convert("RGB")
    description = await describe_image(image)  # Verwende await für die asynchrone Funktion

   # Generiere eine Antwort-Nachricht, als ob der Bot das Bild gerade ansieht
    user_message = (
        f"Der Patient hat dir gerade ein Bild geschickt, tue so als könntest du es wirklich sehen - reagiere NUR auf den Teil ab jetzt:\n"
        f"Ich sehe das Bild vor mir und erkenne folgendes: {description}.\n"
        f"Was fällt dir zu diesen Details auf? Gibt es bestimmte Aspekte des Bildes, die für dich besonders bedeutsam sind?\n"
        f"Ich bin gespannt auf deine Gedanken und Eindrücke zu dem, was du mir geschickt hast.\n"
        f"Verwende keinen Konjuktiv!\n"
        f"Falls du nur eine Person siehst, gehe stehts davon aus dass es der Patient ist."
    )

    gpt3_response = await chat_with_gpt3(user_id, user_message)
    await update.message.reply_text(gpt3_response)



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

async def summarize_descriptions(descriptions: list) -> str:
    combined_descriptions = "\n".join(descriptions)

    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "Du bist die Augen (Sinne) eines anderen Agenten. Der folgende Text ist eine Beschreibung von dem was du siehst, sag was du siehst und fass alles zusammen."},
                {"role": "user", "content": combined_descriptions}
            ],
    )

    summary_response = completion.choices[0].message.content

    return summary_response

async def summarize_website(descriptions: list) -> str:
    combined_descriptions = "\n".join(descriptions)

    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "Du bist die Augen(Sinne) und öffnest eine Website - fasse zusammen was du siehst basierend auf der Beschreibung (halte es sehr kurz)."},
                {"role": "user", "content": combined_descriptions}
            ],
    )

    summary_response = completion.choices[0].message.content

    return summary_response

async def process_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Initialisiere den FER-Detektor
    detector = FER()

    user_id = update.effective_user.id
    file = await update.message.video.get_file()
    video_bytearray = await file.download_as_bytearray()

    # Speichern des Videos in einer temporären Datei
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_file:
        tmp_video_file.write(video_bytearray)
        video_path = tmp_video_file.name

    # Audio aus dem Video extrahieren
    audio_path = extract_audio(video_path)

    # Transkription des Audios
    transcript = recognize_speech_from_path(audio_path)

    # Extrahiere Frames und beschreibe sie
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

    # Fasse die Beschreibungen zusammen
    summary = await summarize_descriptions(descriptions)

    # Öffne das letzte Frame für die Objekterkennung und Emotionserkennung
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Das letzte Frame konnte nicht gelesen werden.")
        os.remove(video_path)
        os.remove(audio_path)
        return

    # Konvertiere das Frame in PIL Image Format für die Objekterkennung
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    detected_objects = [model.config.id2label[label_id.item()] for label_id in results["labels"]]

    # Emotionserkennung auf dem letzten Frame
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

    # Erstelle eine Antwort
    response_text = f"""Bitte reagiere auf das, was du im Video siehst und hörst.
    Der Patient hat dir ein Video geschickt. Du siehst folgendes: {summary}.
    Du erkennst folgende Emotion: {emotion} (Score: {score}). Wenn du nur eine Person siehst - gehe immer davon aus dass es der Patient ist.

    Außerdem verstehst du, was der Patient sagt. Du hörst: {transcript}.

    Beschreibe was du siehst und hörst zurück an den Patienten. Und stelle ihm Fragen dazu als Therapeutin. Reagiere als wärst du
    ein echter Mensch der gerade ein Video von seinem Patienten gesendet bekommen hat und auf den Inhalt reagiert. Du musst nicht auf jedes Detail
    reagieren, gehe davon aus dass das Video vom Patienten kommt und er dir seine Umgebung zeigt. Rede nicht von Szenen oder Video sondern
    sage einfach was du siehst und hörst, z.B. 'Hey das sieht aber nach einer gemuetlichen Wohnung aus, tolle Kueche und Couch - was treibst du da gerade
    spielst du etwa ein Videospiel?'"""

    print("----------------------------------")
    print(response_text)
    print("----------------------------------")

    # Generiere eine Antwort von GPT-3
    gpt3_response = await chat_with_gpt3(user_id, response_text)

    # Sende die Antwort als Sprachnachricht
    await send_voice_message(update, context, gpt3_response)

    # Bereinige temporäre Dateien
    os.remove(video_path)
    os.remove(audio_path)


async def clean_text(text: str, language: str) -> str:
    """
    Tokenizes, removes stop words, and lemmatizes the input text based on the language.

    Parameters:
    text (str): The input text to clean.
    language (str): The language of the text ('en' for English, 'de' for German).

    Returns:
    str: The cleaned text.
    """
    if language == 'en':
        stop_words = set(stopwords.words('english'))
    elif language == 'de':
        stop_words = set(stopwords.words('german'))
    else:
        stop_words = set()  # If language is not supported, don't remove stop words

    lemmatizer = WordNetLemmatizer() if language == 'en' else None

    # Tokenize the text
    tokens = word_tokenize(text.lower()) if language == 'en' else text.lower().split()

    # Remove stop words and non-alphanumeric tokens, then lemmatize if applicable
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words] if lemmatizer else [word for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(cleaned_tokens)

async def extract_text_html(url: str) -> str:
    """
    Extracts and cleans text content from a given URL based on the detected language.

    Parameters:
    url (str): The URL of the webpage from which to extract text.

    Returns:
    str: The cleaned text content from the webpage, or an error message if the URL could not be fetched.
    """
    try:
        # Set headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Make an HTTP GET request
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content
        text_content = soup.get_text(separator=' ')

        # Detect language
        language = detect(text_content)

        # Clean the text content based on detected language
        clean_text_content = await clean_text(text_content, language)

        if len(clean_text_content) > 12500:
            # Optionally, handle very long texts
            pattern = r'\|([^|]+)\|'
            tokens = re.findall(pattern, clean_text_content)
            clean_text_content = " | ".join([word for word in tokens if len(word) < 50])

        return clean_text_content

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error processing content: {e}"


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_text = update.message.text
    urls = extract_urls(message_text)

    if urls:
        responses = []
        for url in urls:
            screenshot_path = '/path/to/screenshot.png'
            summary = await extract_text_html(url)
            response = await generate_response_from_text(summary)
            responses.append(f"URL: {url}\nResponse: {response}")
            print("URL FOUND")

        # Send back the responses
        await update.message.reply_text("\n\n".join(responses))
    else:
        # Handle regular messages
        await update.message.reply_text("No URLs found in the message.")


# Define a function to extract URLs from a message
def extract_urls(message: str) -> list:
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(message)



if __name__ == '__main__':
    application = ApplicationBuilder().token(telegram_token).build()

    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    voice_handler = MessageHandler(filters.VOICE, process_voice_message)
    video_handler = MessageHandler(filters.VIDEO, process_video_message)
    reset_handler = CommandHandler('reset', reset)
    photo_handler = MessageHandler(filters.PHOTO, process_photo)


    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(voice_handler)
    application.add_handler(video_handler)
    application.add_handler(reset_handler)
    application.add_handler(photo_handler)

    application.run_polling()
