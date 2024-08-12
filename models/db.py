import aiomysql
from config import config
import stripe
import json

async def get_mysql_connection():
    return await aiomysql.connect(
        host=config["MYSQL_HOST"],
        user=config["MYSQL_USER"],
        password=config["MYSQL_PASSWORD"],
        db=config["MYSQL_DATABASE"],
    )

async def get_conversation_history(user_id):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("SELECT history FROM conversations WHERE user_id = %s", (user_id,))
            result = await cursor.fetchone()
            return json.loads(result['history']) if result and result['history'] else []
    finally:
        connection.close()

# Function to get message count for a user
async def get_message_count(user_id):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("SELECT message_count FROM user_message_count WHERE user_id = %s", (user_id,))
            result = await cursor.fetchone()
            # Return None if no result or message_count is not found
            return result['message_count'] if result and 'message_count' in result else 0
    finally:
        connection.close()

# Function to update message count for a user
async def update_message_count(user_id):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor() as cursor:
            current_count = await get_message_count(user_id)
            new_count = current_count + 1  # Increment the count

            if current_count > 0:
                # Update existing record
                await cursor.execute("""
                    UPDATE user_message_count
                    SET message_count = %s
                    WHERE user_id = %s
                """, (new_count, user_id))
            else:
                # Insert new record with count starting at 1
                await cursor.execute("""
                    INSERT INTO user_message_count (user_id, message_count)
                    VALUES (%s, %s)
                """, (user_id, new_count))

            await connection.commit()
            return new_count
    finally:
        connection.close()

async def update_conversation_history(user_id, new_history):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor() as cursor:
            new_history_json = json.dumps(new_history)
            await cursor.execute("""
                INSERT INTO conversations (user_id, history)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE history = VALUES(history)
            """, (user_id, new_history_json))
            await connection.commit()
    finally:
        connection.close()

async def reset_conversation(user_id):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM message_store WHERE session_id = %s", (user_id,))
        await connection.commit()
    finally:
        connection.close()

# Function to retrieve Stripe customer ID from payments table
async def get_stripe_customer_id(user_id):
    connection = await get_mysql_connection()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("""
                SELECT stripe_customer_id FROM payments WHERE user_id = %s
            """, (user_id,))
            result = await cursor.fetchone()
            return result['stripe_customer_id'] if result and result['stripe_customer_id'] else None
    finally:
        connection.close()

# Function to get subscription status
async def get_subscription_status(user_id):
    stripe_customer_id = await get_stripe_customer_id(user_id)
    if not stripe_customer_id:
        return "No Stripe customer ID found for this user."

    try:
        subscriptions = stripe.Subscription.list(customer=stripe_customer_id)

        if subscriptions.data:
            subscription = subscriptions.data[0]
            status = subscription.status
            return status
        else:
            return "No subscriptions found for this customer."

    except stripe.error.StripeError as e:
        return f"Stripe API error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
