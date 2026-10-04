import base64
import hashlib
import hmac
import logging
from xml.sax.saxutils import escape
from fastapi import APIRouter, Request, Form, HTTPException, Response

from app.config import settings
from app.gateway.pipeline import gateway_pipeline

logger = logging.getLogger("channel.whatsapp")

router = APIRouter()


def create_twiml_response(message_body: str) -> Response:
    """Generate standard TwiML XML response for Twilio WhatsApp messages."""
    safe_body = escape(message_body)
    xml_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Response>\n'
        f'    <Message>{safe_body}</Message>\n'
        f'</Response>'
    )
    return Response(content=xml_content, media_type="application/xml")


def verify_twilio_signature(request: Request, form_data: dict) -> bool:
    """
    Validate Twilio webhook signature when TWILIO_AUTH_TOKEN is configured.
    In development mode, verification is skipped if no token is configured.
    In non-development environments, unconfigured tokens fail closed to prevent spoofing.
    """
    auth_token = settings.TWILIO_AUTH_TOKEN
    if not auth_token:
        if settings.is_development:
            return True
        logger.error(
            "Twilio signature verification rejected: TWILIO_AUTH_TOKEN not configured in non-development environment."
        )
        return False

    signature = request.headers.get("X-Twilio-Signature", "")
    signed_payload = str(request.url) + "".join(
        f"{key}{form_data[key]}" for key in sorted(form_data)
    )
    digest = hmac.new(
        auth_token.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha1
    ).digest()
    expected_signature = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...)
):
    """
    Twilio WhatsApp Webhook Endpoint.
    Acts as a thin adapter routing WhatsApp messages through the AI Safety Gateway.
    """
    form_data = {"Body": Body, "From": From}
    if not verify_twilio_signature(request, form_data):
        logger.warning("Rejected WhatsApp webhook with invalid Twilio signature.")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature.")

    try:
        sender_phone = From.replace("whatsapp:", "").strip()
        user_message = Body.strip()

        logger.info(
            "Incoming WhatsApp message received from masked sender %s***",
            sender_phone[:4]
        )

        # Process through AI Safety Gateway
        gateway_res = gateway_pipeline.process(
            user_id=sender_phone,
            message=user_message,
            channel="whatsapp"
        )

        return create_twiml_response(gateway_res.response)

    except Exception as e:
        logger.error("Error handling WhatsApp webhook: %s", e)
        fallback_msg = (
            "Sorry, our healthcare assistant is experiencing technical difficulties. "
            "Please try again shortly."
        )
        return create_twiml_response(fallback_msg)

