"""
This package contains models and functions for data processing.
"""
from .db import get_mysql_connection, get_conversation_history, update_conversation_history, reset_conversation, get_stripe_customer_id, get_subscription_status, get_message_count, update_message_count
from .gpt import generate_response_gpt
from .audio_processing import extract_audio, send_voice_message, text_to_speech_conversion, speech_to_text_conversion
from .image_processing import describe_image, extract_frames_from_video, summarize_descriptions
