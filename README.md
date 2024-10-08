# AI Therapist Chatbot for Telegram

This repository hosts an AI-driven therapist chatbot for Telegram, supporting both German and English. The chatbot handles text, images, voice messages, video messages, and URLs, and can respond in text or voice. It's built with advanced AI technologies, including OpenAI's GPT (for text generation), Whisper (for speech recognition), and TTS1 (for text-to-speech). The chatbot also features a Retrieval-Augmented Generation (RAG) system and integrates Stripe for payment processing.

## Features

- **Multilingual Support**: German and English.
- **Multi-Modal Input**: Text, images, voice, video, URLs.
- **Response Modes**: Text and voice (using OpenAI TTS1).
- **RAG System**: Enhanced response accuracy via retrieved data.
- **OpenAI Integration**: GPT for text, Whisper for voice-to-text, TTS1 for text-to-speech.
- **Stripe Integration**: Handles payments for premium features.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/jonasge1992/ai_therapist_german.git
    cd ai_therapist_german
    ```

2. Install dependencies:
    ```bash
    pyenv install 3.10.6
    pyenv local 3.10.6
    pip install -r requirements.txt
    ```

3. Configure environment variables:
    ```bash
    cp .env.example .env
    # Add your API keys and configuration details
    ```

4. Run the bot:
    ```bash
    python main.py
    ```
