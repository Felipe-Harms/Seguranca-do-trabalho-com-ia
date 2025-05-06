# src/notifier.py

import os
from twilio.rest import Client
from dotenv import load_dotenv

# Carrega as variáveis do .env
load_dotenv()

def send_sms(body):
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE")
    to_number = os.getenv("PHONE_TO_NOTIFY")

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=body,
        from_=from_number,
        to=to_number
    )
    print(f"[✔] SMS enviado! SID: {message.sid}")
