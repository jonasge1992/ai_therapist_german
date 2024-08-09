from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, PreCheckoutQueryHandler
)
from config import config
import stripe
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
import ffmpeg



# Function to start interaction with payment
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stripe.api_key = config["STRIPE_API_KEY"]
    user_id = update.effective_user.id

    welcome_message = f"""Um das Gespräch fortzusetzen: Drücke unten auf [ Gespräch fortsetzen ]

🧠 Der erste KI Life Coach, der 24/7 verfügbar ist und von Tausenden von Menschen genutzt wird

🧑‍⚖️ Ich bin ein KI-Lebenscoach. Ich bin kein lizenzierter Psychologe, Therapeut oder Gesundheitsberater
und ersetze diese nicht. Ich übernehme keine Verantwortung für die Ergebnisse deiner Handlungen und
jeglichen Schaden, den du aufgrund der Nutzung oder Nichtnutzung der verfügbaren Informationen
erleiden könntest. Nutze dein Urteil und deine Sorgfaltspflicht, bevor du eine der vorgeschlagenen
Maßnahmen oder Pläne umsetzt. Nutze mich nicht, wenn du dich oder andere in Gefahr siehst,
sondern wende dich in diesem Fall an professionelle Hilfe
https://www.suizidprophylaxe.de/hilfsangebote/hilfsangebote/

✅ Ändere deine negativen Gedanken 💭.
✅ Werden aktiv und löse dich aus der Sackgasse 💥
✅ Mach dich fit, das hilft deinem Geist 🏋️
✅ Sprich mit mir über deinen Tag 🗣️
✅ Fühle dich besser, indem du nach dir schaust 🤗

Du kannst
🗣️ Sprachnachricht schicken, ich antworte per Stimme
🤳 Videonachricht schicken, ich antworte per Stimme
💬 Chat-Nachricht schicken, ich antworte per Text
📸 Foto schicken und wir reden darüber
🔎 URL schicken und wir diskutieren es

Schreiben jederzeit /reset, um den gesamten Konversationsverlauf von unseren Servern zu löschen

💡 Rückmeldung
Hast du Feedback, Ideen und Fehler für mich?
https://jeetah.canny.io/therapie-ki"""

    try:
        # Create a Checkout Session
        session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': 'price_1PlvCNC09XxLfCPVSWRgZMMC',  # Use the price ID for the product
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url='https://stripe-webhook-1-e83deee71032.herokuapp.com/order/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://stripe-webhook-1-e83deee71032.herokuapp.com/cancel',
            subscription_data={
                'metadata': {
                    'user_id': user_id,  # Store user ID in metadata
                }
            }
        )

        # Create an inline keyboard with the payment button
        keyboard = [[InlineKeyboardButton("✨ Gespräch fortsetzen", url=session.url)]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send message with inline button
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_message,
            reply_markup=reply_markup,
            disable_web_page_preview=True
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f'There was an issue creating the payment. Please try again later. {e}'
        )
