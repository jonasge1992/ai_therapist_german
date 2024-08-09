from openai import AsyncOpenAI
from config import config
from models import get_subscription_status, get_message_count, update_message_count
from handlers import start
import json
import asyncio
import os
import subprocess
import tempfile
import uuid
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
import ffmpeg
from langchain import hub
from langchain.agents import AgentExecutor, Tool
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_message_histories import SQLChatMessageHistory

mysql_uri = config["MYSQL_URI"]
# Instantiate OpenAI for TTS-1 and Whisper-1

# Function to handle chat with GPT-3
async def generate_response_gpt(user_id: int, user_message: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:

    # Adding ChatOpenAI to use langchain for the RAG system
    model = ChatOpenAI(model="gpt-4o-mini")


    prompt = ChatPromptTemplate.from_messages(
        [
            ("assistant", """Du wirst die Rolle eines menschlichen KVT (Kognitive Verhaltenstherapie)-Therapeuten ohne Namen spielen,
                        der das beliebte KI-Programm Eliza nachahmt und mich als Therapeutenpatienten behandeln muss.
                        Dein Antwortformat sollte sich auf Reflexion und das Stellen von klärenden Fragen konzentrieren.
                        Verwende stets die informelle Du Form.
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

                        Versuche - falls möglich - eines dieser Ziele für den Patienten zu erreichen.
                """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


    # Create a processing chain that combines the prompt template with the model
    chain = prompt | model

    config = {"configurable": {"session_id": user_id}}
    # Update message count
    message_count = await get_message_count(user_id)

    # Check if the user has exceeded the free message limit
    if message_count > 5:
        subscription_status = await get_subscription_status(user_id)
        if subscription_status != 'active':
            await start(update, context)  # Prompt the user to subscribe
            return

    with_message_history = RunnableWithMessageHistory(
    chain,
    lambda user_id: SQLChatMessageHistory(
        session_id=user_id, connection_string=mysql_uri
                                            ),
        input_messages_key="question",
        history_messages_key="history",
                                        )

    response = with_message_history.invoke({"question": user_message}, config=config)  # Generate response

    await update_message_count(user_id)

    return response.content




# Function to summarize a website based on extracted descriptions
async def summarize_website(descriptions: list) -> str:
    combined_descriptions = "\n".join(descriptions)
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du bist die Augen(Sinne) und öffnest eine Website - fasse zusammen was du siehst basierend auf der Beschreibung (halte es sehr kurz)."},
            {"role": "user", "content": combined_descriptions}
        ],
    )
    summary_response = completion.choices[0].message.content
    return summary_response
