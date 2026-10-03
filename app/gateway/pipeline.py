import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

from app.gateway.input_guard import InputGuard
from app.gateway.output_guard import OutputGuard
from app.gateway.fallback import FallbackHandler
from app.ai.rag import rag_engine, MedicalRAG
from app.ai.llm import get_llm_client, BaseLLMClient
from app.observability.tracer import tracer, GatewayTracer, TraceRecord

logger = logging.getLogger("gateway.pipeline")


@dataclass
class GatewayResponse:
    """Standardized response returned by AI Safety Gateway."""
    response: str
    trace_id: str
    status: str
    pii_redacted: Dict[str, int]
    model_used: str
    tokens_used: int
    duration_ms: float
    disclaimer_appended: bool
    is_fallback: bool = False


class GatewayPipeline:
    """
    Core AI Safety Gateway Orchestrator.
    
    Data Flow:
    1. Start trace timer with unique Trace ID.
    2. Input Guard: Validate, detect prompt injection, redact PII.
    3. AI Orchestrator: Retrieve medical context (RAG) and call LLM.
    4. Output Guard: Check leaked PII, safety rules, format, append disclaimer.
    5. Fallback Router: If any stage fails, cleanly degrade to safe response.
    6. Record telemetry trace (without logging sensitive PII).
    """

    def __init__(
        self,
        input_guard: Optional[InputGuard] = None,
        output_guard: Optional[OutputGuard] = None,
        rag: Optional[MedicalRAG] = None,
        llm: Optional[BaseLLMClient] = None,
        tracer_inst: Optional[GatewayTracer] = None
    ):
        self.input_guard = input_guard or InputGuard()
        self.output_guard = output_guard or OutputGuard()
        self.rag = rag or rag_engine
        self.llm = llm or get_llm_client()
        self.tracer = tracer_inst or tracer

    def process(
        self,
        user_id: str,
        message: str,
        channel: str = "api",
        user_profile: Optional[Dict] = None
    ) -> GatewayResponse:
        """Process a request through the entire safety gateway."""
        trace = self.tracer.start_trace(user_id=user_id, channel=channel)
        pipeline_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. INPUT GUARD
        # -------------------------------------------------------------
        res_in = self.input_guard.validate(message)
        trace.input_guard_ms = res_in.duration_ms
        trace.pii_redacted_count = sum(res_in.pii_redacted.values())
        trace.sanitized_preview = res_in.sanitized_message[:80] if res_in.sanitized_message else ""

        if not res_in.passed:
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            trace.total_ms = round(total_ms, 2)

            if res_in.is_injection:
                fallback_msg = FallbackHandler.get_fallback_response(
                    reason=res_in.rejection_reason or "Injection",
                    is_injection=True
                )
                trace.status = "REJECTED_INJECTION"
                trace.fallback_invoked = True
                self.tracer.record_trace(trace)
                return GatewayResponse(
                    response=fallback_msg,
                    trace_id=trace.trace_id,
                    status="REJECTED_INJECTION",
                    pii_redacted=res_in.pii_redacted,
                    model_used="none",
                    tokens_used=0,
                    duration_ms=trace.total_ms,
                    disclaimer_appended=False,
                    is_fallback=True
                )
            else:
                trace.status = "REJECTED_INPUT"
                self.tracer.record_trace(trace)
                return GatewayResponse(
                    response=res_in.rejection_reason or "Invalid request input.",
                    trace_id=trace.trace_id,
                    status="REJECTED_INPUT",
                    pii_redacted=res_in.pii_redacted,
                    model_used="none",
                    tokens_used=0,
                    duration_ms=trace.total_ms,
                    disclaimer_appended=False,
                    is_fallback=False
                )

        # -------------------------------------------------------------
        # 2. AI ORCHESTRATOR - RAG RETRIEVAL
        # -------------------------------------------------------------
        t_rag = time.perf_counter()
        retrieved_docs = self.rag.retrieve(res_in.sanitized_message)
        trace.rag_ms = round((time.perf_counter() - t_rag) * 1000, 2)

        prompt = self.rag.build_prompt(
            query=res_in.sanitized_message,
            retrieved_docs=retrieved_docs,
            user_profile=user_profile
        )

        # -------------------------------------------------------------
        # 3. AI ORCHESTRATOR - LLM INVOCATION
        # -------------------------------------------------------------
        t_llm = time.perf_counter()
        try:
            llm_res = self.llm.generate(prompt=prompt)
            trace.llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            trace.model_used = llm_res.model
            trace.tokens_used = llm_res.tokens_used
            raw_output = llm_res.text
        except Exception as e:
            logger.error(f"LLM failure on request {trace.trace_id}: {e}")
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            trace.total_ms = round(total_ms, 2)
            trace.status = "FALLBACK_SYSTEM"
            trace.fallback_invoked = True
            self.tracer.record_trace(trace)

            fallback_msg = FallbackHandler.get_fallback_response("LLM call failed")
            return GatewayResponse(
                response=fallback_msg,
                trace_id=trace.trace_id,
                status="FALLBACK_SYSTEM",
                pii_redacted=res_in.pii_redacted,
                model_used="none",
                tokens_used=0,
                duration_ms=trace.total_ms,
                disclaimer_appended=False,
                is_fallback=True
            )

        # -------------------------------------------------------------
        # 4. OUTPUT GUARD
        # -------------------------------------------------------------
        res_out = self.output_guard.validate(raw_output)
        trace.output_guard_ms = res_out.duration_ms

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        trace.total_ms = round(total_ms, 2)

        if not res_out.passed:
            trace.status = "FALLBACK_SAFETY"
            trace.fallback_invoked = True
            self.tracer.record_trace(trace)

            fallback_msg = FallbackHandler.get_fallback_response(
                reason=res_out.rejection_reason or "Output safety",
                is_unsafe=res_out.is_unsafe
            )
            return GatewayResponse(
                response=fallback_msg,
                trace_id=trace.trace_id,
                status="FALLBACK_SAFETY",
                pii_redacted=res_in.pii_redacted,
                model_used=trace.model_used,
                tokens_used=trace.tokens_used,
                duration_ms=trace.total_ms,
                disclaimer_appended=False,
                is_fallback=True
            )

        # -------------------------------------------------------------
        # 5. SUCCESS & TRACE COMPLETION
        # -------------------------------------------------------------
        trace.status = "SUCCESS"
        self.tracer.record_trace(trace)

        return GatewayResponse(
            response=res_out.final_text,
            trace_id=trace.trace_id,
            status="SUCCESS",
            pii_redacted=res_in.pii_redacted,
            model_used=trace.model_used,
            tokens_used=trace.tokens_used,
            duration_ms=trace.total_ms,
            disclaimer_appended=res_out.disclaimer_added,
            is_fallback=False
        )


# Singleton pipeline instance
gateway_pipeline = GatewayPipeline()
