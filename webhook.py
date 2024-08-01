import json
import os
import stripe
from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_API_KEY")  # Replace with your Stripe secret key
endpoint_secret = os.getenv("STRIPE_ENDPOINT_SECRET_KEY")  # Replace with your Stripe webhook secret

# MongoDB configuration
mongo_uri = os.getenv("MONGO_DB_URI")  # Replace with your MongoDB URI
database_name = "telegram_bot"  # Replace with your database name
collection_name = "payments"  # Replace with your collection name

app = Flask(__name__)

# Initialize MongoDB client and collection
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

@app.route('/webhook', methods=['POST'])
def webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        print(f"Invalid payload: {e}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        print(f"Invalid signature: {e}")
        return jsonify({'error': 'Invalid signature'}), 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        subscription_id = session.get('subscription')

        if subscription_id:
            subscription = stripe.Subscription.retrieve(subscription_id)
            user_id = subscription.metadata.get('user_id', 'unknown')
            tguser = subscription.metadata.get('tguser', 'unknown')

            print(f"Subscription succeeded, User ID: {user_id}, TG User ID: {tguser}")

            # Update MongoDB with additional fields
            update_payment_status(
                subscription_id,  # payment_id
                subscription.items.data[0].price.unit_amount / 100,  # amount_received (convert from cents)
                subscription.currency,  # currency
                user_id,
                tguser
            )

    return jsonify(success=True)

def update_payment_status(payment_id, amount_received, currency, user_id, tguser):
    # Create a filter to find the document that needs to be updated
    filter = {"payment_id": payment_id}

    # Create an update document
    update = {
        "$set": {
            "amount_received": amount_received,
            "currency": currency,
            "status": "succeeded",
            "user_id": user_id,  # Store user ID in the database
            "tguser": tguser,  # Store Telegram user ID in the database
            "updated_at": datetime.utcnow()
        }
    }

    # Perform the update operation
    result = collection.update_one(filter, update, upsert=True)

    if result.modified_count > 0:
        print(f"Successfully updated payment {payment_id} for user {user_id}")
    else:
        print(f"No update performed for payment {payment_id}")

if __name__ == '__main__':
    app.run(port=4221)  # Run Flask app on port 4221
