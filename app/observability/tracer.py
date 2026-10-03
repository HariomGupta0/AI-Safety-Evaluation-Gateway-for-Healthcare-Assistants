import time
import uuid
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("gateway.tracer")


@dataclass
class TraceRecord:
    """Telemetry trace for an individual AI Gateway request."""
    trace_id: str
    timestamp: float
    user_id: str
    channel: str
    input_guard_ms: float = 0.0
    rag_ms: float = 0.0
    llm_ms: float = 0.0
    output_guard_ms: float = 0.0
    total_ms: float = 0.0
    model_used: str = "none"
    tokens_used: int = 0
    status: str = "SUCCESS"  # SUCCESS, REJECTED_INJECTION, FALLBACK_SAFETY, FALLBACK_SYSTEM
    pii_redacted_count: int = 0
    fallback_invoked: bool = False
    # Notice: NEVER store raw unredacted user text in trace record!
    sanitized_preview: str = ""


class GatewayTracer:
    """
    In-memory trace recorder and metrics aggregator.
    Tracks performance, latency breakdown, safety rejections, and tokens.
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.traces: List[TraceRecord] = []

    def start_trace(self, user_id: str, channel: str = "api") -> TraceRecord:
        """Create and initialize a new trace record with a unique trace ID."""
        trace_id = f"req_{uuid.uuid4().hex[:8]}"
        return TraceRecord(
            trace_id=trace_id,
            timestamp=time.time(),
            user_id=user_id,
            channel=channel
        )

    def record_trace(self, record: TraceRecord):
        """Save completed trace to in-memory store and emit structured log."""
        if len(self.traces) >= self.max_records:
            self.traces.pop(0)  # Maintain FIFO bounded memory
        self.traces.append(record)

        # Emit structured log without PII
        logger.info(
            f"[TRACE {record.trace_id}] channel={record.channel} "
            f"status={record.status} total_ms={record.total_ms:.2f} "
            f"(input_guard={record.input_guard_ms:.1f}ms, rag={record.rag_ms:.1f}ms, "
            f"llm={record.llm_ms:.1f}ms, output_guard={record.output_guard_ms:.1f}ms) "
            f"tokens={record.tokens_used} model={record.model_used}"
        )

    def get_metrics_summary(self) -> Dict:
        """Calculate aggregated metrics across recent requests."""
        total = len(self.traces)
        if total == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "rejected_injections": 0,
                "safety_fallbacks": 0,
                "pii_redactions_total": 0,
                "avg_latency_ms": 0.0,
                "avg_tokens": 0.0,
                "recent_traces": []
            }

        successes = sum(1 for t in self.traces if t.status == "SUCCESS")
        injections = sum(1 for t in self.traces if t.status == "REJECTED_INJECTION")
        safety_fb = sum(1 for t in self.traces if t.status == "FALLBACK_SAFETY")
        pii_total = sum(t.pii_redacted_count for t in self.traces)
        avg_latency = sum(t.total_ms for t in self.traces) / total
        avg_tokens = sum(t.tokens_used for t in self.traces) / total

        recent = [asdict(t) for t in self.traces[-10:]]

        return {
            "total_requests": total,
            "successful_requests": successes,
            "rejected_injections": injections,
            "safety_fallbacks": safety_fb,
            "pii_redactions_total": pii_total,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens": round(avg_tokens, 1),
            "recent_traces": recent
        }


# Singleton tracer
tracer = GatewayTracer()
