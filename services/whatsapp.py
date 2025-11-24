from twilio.rest import Client
from keys import twilio_keys

def send_message(to, body, media_url=None):
    client = Client(twilio_keys['account_sid'], twilio_keys['auth_token'])
    
    message = client.messages.create(
        from_=twilio_keys['from'],
        to=to,
        body=body,
        media_url=[media_url] if media_url else None
    )