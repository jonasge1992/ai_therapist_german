import soundfile as sf
import librosa
from telegram import Update
from telegram.ext import ContextTypes
import io
import boto3
import os
import tempfile
from pydub import AudioSegment
import logging
import speech_recognition as sr
from langdetect import detect
from openai import OpenAI
from telegram.ext import CallbackContext
from config import config
# Function to extract audio from video
from moviepy.editor import VideoFileClip
import uuid
import ffmpeg


def speech_to_text_conversion(file_path):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    # Open the audio file specified by file_path in binary read mode
    with open(file_path, 'rb') as file_like:
        # Use OpenAI's Whisper-1 model to convert speech in the audio file to text
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=file_like
        )
    # Return the transcribed text from the audio file
    return transcription.text



def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip.audio.write_audiofile(audio_path)
    return audio_path

async def text_to_speech_conversion(text) -> str:
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    # Generate a unique ID for temporary file names
    unique_id = uuid.uuid4().hex
    mp3_path = f'{unique_id}.mp3'  # Path for temporary MP3 file
    ogg_path = f'{unique_id}.ogg'  # Path for final OGG file

    # Convert the input text to speech and save it as an MP3 file
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",       # Use the text-to-speech model
        voice="nova",        # Specify the voice model to use
        input=text           # Text to convert to speech
    ) as response:
        # Write the streamed audio data to the MP3 file
        with open(mp3_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    # Convert the MP3 file to OGG format with OPUS codec using ffmpeg
    ffmpeg.input(mp3_path).output(ogg_path, codec='libopus').run(overwrite_output=True)

    # Remove the temporary MP3 file as it is no longer needed
    os.remove(mp3_path)

    # Return the path to the final OGG file
    return ogg_path

async def send_voice_message(update: Update, context: CallbackContext, audio_path: str):
    # Open the audio file and send it as a voice message
    with open(audio_path, 'rb') as audio_data:
        await update.message.reply_voice(voice=audio_data)

    # Remove the OGG file from the server after sending it
    if os.path.exists(audio_path):
        os.remove(audio_path)
