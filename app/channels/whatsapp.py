import logging
from xml.sax.saxutils import escape
from fastapi import APIRouter, Request, Form
from fastapi.responses import PlainTextResponse

from app.gateway.pipeline import gateway_pipeline

logger = logging.getLogger("channel.whatsapp")

router = APIRouter()


def create_twiml_response(message_body: str) -> str:
    """Generate standard TwiML XML to respond to incoming Twilio WhatsApp messages."""
    safe_body = escape(message_body)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Response>\n'
        f'    <Message>{safe_body}</Message>\n'
        f'</Response>'
    )


@router.post("/webhook", response_class=PlainTextResponse)
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    """
    Twilio WhatsApp Webhook Endpoint.
    Acts as a thin adapter routing WhatsApp messages through the AI Safety Gateway.
    """
    try:
        sender_phone = From.replace("whatsapp:", "").strip()
        user_message = Body.strip()

        logger.info(f"Incoming WhatsApp message from {sender_phone[:4]}***: {user_message[:40]}...")

        # Process through AI Safety Gateway
        gateway_res = gateway_pipeline.process(
            user_id=sender_phone,
            message=user_message,
            channel="whatsapp"
        )

        return create_twiml_response(gateway_res.response)

    except Exception as e:
        logger.error(f"Error handling WhatsApp webhook: {e}")
        fallback_msg = (
            "Sorry, our healthcare assistant is experiencing technical difficulties. "
            "Please try again shortly."
        )
        return create_twiml_response(fallback_msg)
