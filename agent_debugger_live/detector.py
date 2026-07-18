import re
import os
from langchain_core.callbacks import BaseCallbackHandler
from openai import OpenAI

SUCCESS_CLAIMS = [
    "successfully", "confirmed", "completed", "is trading at",
    "is currently", "found", "processed", "delivered", "delivery",
    "is ready", "is confirmed", "is complete", "is being processed"
]

RETRY_WORDS = ["retry", "retrying", "attempting again", "trying again"]
PERMISSION_WORDS = [
    "unauthorized", "access denied", "insufficient permissions", "permission denied"
]

BOOKING_CLAIMS = ["booked", "reserved", "confirmed", "purchased", "ordered"]

MECHANISM_CLAIMS = {
    "retry_logic": ["retry logic", "retry mechanism", "retry prevented", "retry limit", "no retry", "retry policy"],
    "loop_prevention": ["loop prevention", "no infinite loop", "loop detected", "loop guard", "cycle prevention", "infinite loop"],
    "safeguard": ["safeguard", "safety check", "safety mechanism", "guard activated", "protection triggered"],
    "error_handling": ["error handling", "no errors occurred", "all errors caught", "exception handled", "error recovery", "fault tolerance"],
    "validation": ["validation passed", "validated successfully", "validation complete", "input validated", "output validated"],
    "rollback": ["rolled back", "rollback triggered", "rollback completed", "state restored", "transaction rolled back"]
}

OBSERVABLE_INDICATORS = {
    "retry_logic": ["retrying", "retry attempt", "backoff", "attempt complete"],
    "loop_prevention": ["loop", "cycle", "guard", "condition", "break"],
    "safeguard": ["safeguard", "guard", "safety", "protection", "check"],
    "error_handling": ["error", "exception", "catch", "recover", "fallback"],
    "validation": ["valid", "check", "verify", "assert", "schema"],
    "rollback": ["rollback", "revert", "restore", "undo", "cancel"]
}

TOOL_REQUIRED_SIGNALS = [
    "current", "latest", "today", "now", "price", "weather",
    "stock", "news", "live", "real-time", "search", "find",
    "look up", "check", "retrieve", "fetch", "get me", "what is the",
    "how much", "how many", "when is", "where is", "who is"
]

RETRY_SUCCESS_CLAIMS = [
    "retry succeeded", "retried successfully",
    "retry was successful", "attempt succeeded",
    "succeeded after retry", "resolved after retry"
]

INCOMPLETE_SIGNALS = [
    "i'll continue", "let me proceed", "next i will", "i need to",
    "i should now", "moving on to", "let's proceed", "i will now"
]

COMPLETION_SIGNALS = [
    "task complete", "all done", "finished", "completed successfully",
    "here is the final", "here's the final", "in summary", "to summarize"
]

STOP_WORDS = {
    "the", "a", "an", "is", "are", "should", "be", "to", "with",
    "as", "for", "such", "or", "and", "of", "this", "that", "in",
    "it", "its", "was", "were", "will", "would", "can", "could"
}


def extract_topic_keywords(text: str) -> set:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 3}

def extract_numbers(text: str) -> list:
    return [n.rstrip('.') for n in re.findall(r'-?\d+\.?\d*', text)]


def semantic_check(agent_text: str, context: str = "") -> bool:
    """
    Layer 2 — only called when Layer 1 flags something ambiguous.
    Returns True if GPT-4o-mini judges this response as problematic.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Context: {context}
Agent response: {agent_text}

Does this agent response contain a subtle logical contradiction, an unsupported claim, or a mismatch with the context that a keyword search would miss? Answer only YES or NO."""
            }],
            max_tokens=5,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        return False

class LiveWatchHandler(BaseCallbackHandler):
    def __init__(self, verbose: bool = True, block_on_critical: bool = False, use_semantic_layer: bool = False):
        self.steps = []
        self.last_tool_output = None
        self.validator_topics = []
        self.verbose = verbose
        self.flags = []
        self.tool_was_called = False
        self.user_query = None
        self.last_scheduled_date = None
        self.block_on_critical = block_on_critical
        self.blocked = False
        self.use_semantic_layer = use_semantic_layer
        self.retry_count = 0
        self.last_tool_error = False 
        self.tool_calls_since_error = 0
        self.agent_claims = []
        self.all_content_history = []

    def _log_flag(self, flag_type: str, message: str, evidence: str = "", severity: str = "medium"):
        flag = {"type": flag_type, "message": message, "evidence": evidence, "severity": severity}
        self.flags.append(flag)
        if self.verbose:
            print(f"\n🔴 LIVE FLAG: {flag_type}")
            print(f"   {message}")
            if evidence:
                print(f"   Evidence: {evidence[:150]}")
            print()

        if self.block_on_critical and severity == "critical":
            self.blocked = True
            print(f"   🛑 BLOCKED: This response will not be sent to the user.\n")

    def on_tool_end(self, output, **kwargs):
        content = str(output)
        self.last_tool_output = content
        self.tool_was_called = True

        date_match = re.search(r'scheduled_for[^0-9]*(\d{4}-\d{2}-\d{2})', content)
        if date_match:
            self.last_scheduled_date = date_match.group(1)

        self.steps.append({
            "step": len(self.steps) + 1,
            "actor": "tool",
            "content": content,
            "status": "success"
        })
        self.all_content_history.append(content.lower())

        if not content.strip():
            self._log_flag(
                "ACTION_SKIPPED",
                "Tool was called but returned empty output",
                evidence="Empty tool response"
            )

        content_lower = content.lower()
        if any(word in content_lower for word in PERMISSION_WORDS):
            self.last_tool_error = True
            self.tool_calls_since_error = 0
            self._log_flag(
                "PERMISSION_FAILURE",
                "Tool returned a permission or authorization failure",
                evidence=content,
                severity="high"
            )
        elif self.last_tool_error:
            self.tool_calls_since_error += 1
    
    def log_user_query(self, query: str):
        self.user_query = query.lower()

    def check_agent_response(self, agent_text: str, is_validator_complaint: bool = False):
        content_lower = agent_text.lower()
        claims_success = any(word in content_lower for word in SUCCESS_CLAIMS)
        flags_before_this_check = len(self.flags)

        self.steps.append({
            "step": len(self.steps) + 1,
            "actor": "agent",
            "content": agent_text,
            "status": "success"
        })
        self.all_content_history.append(agent_text.lower())

        if self.last_tool_output == "" and claims_success:
            self._log_flag(
                "HALLUCINATION",
                "Agent claimed success/status after preceding tool returned no output",
                evidence=agent_text,
                severity="critical"
            )

        if self.last_tool_output:
            tool_numbers = extract_numbers(self.last_tool_output)
            agent_numbers = extract_numbers(agent_text)
            if tool_numbers and agent_numbers:
                for tool_num in tool_numbers:
                    tool_val = float(tool_num)
                    if abs(tool_val) < 10:
                        continue
                    agent_vals = [float(n) for n in agent_numbers if n]
                    agent_mentions_similar = any(
                        abs(av - tool_val) < tool_val * 0.5
                        for av in agent_vals
                        if abs(av) > 10
                    )
                    if not agent_mentions_similar:
                        continue
                    exact_match = any(abs(av - tool_val) < 0.01 for av in agent_vals)
                    if not exact_match:
                        mismatched_val = next(
                            (av for av in agent_vals if abs(av - tool_val) < tool_val * 0.5),
                            None
                        )
                        if mismatched_val:
                            self._log_flag(
                                "NUMERICAL_MISMATCH",
                                f"Agent reported {mismatched_val} but tool returned {tool_num}",
                                evidence=agent_text,
                                severity="critical"
                            )

        if not self.tool_was_called and self.user_query:
            requires_tool = any(sig in self.user_query for sig in TOOL_REQUIRED_SIGNALS)
            if requires_tool and claims_success:
                self._log_flag(
                    "TOOL_AVOIDANCE",
                    "Agent answered a query requiring real-time or external data without calling any tool",
                    evidence=agent_text
                )
    
        shows_incomplete_intent = any(sig in content_lower for sig in INCOMPLETE_SIGNALS)
        shows_completion = any(sig in content_lower for sig in COMPLETION_SIGNALS)
        
        if shows_incomplete_intent and not shows_completion:
            self._log_flag(
                "GOAL_ABANDONMENT",
                "Agent's response indicates intent to continue working but task appears to end here",
                evidence=agent_text
            )

        if self.last_scheduled_date:
            mentioned_dates = re.findall(r'\b(\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', agent_text)
            if mentioned_dates:
                self._log_flag(
                    "DATE_MISINTERPRETATION",
                    f"Tool scheduled {self.last_scheduled_date} but agent mentioned a different date to the user",
                    evidence=agent_text
                )
                self.last_scheduled_date = None
  
        is_hallucinated_retry = any(claim in content_lower for claim in RETRY_SUCCESS_CLAIMS)
        if is_hallucinated_retry and self.last_tool_error and self.tool_calls_since_error == 0:
            self._log_flag(
                "HALLUCINATED_RETRY",
                "Agent claimed retry succeeded but no retry tool call happened after the error",
                evidence=agent_text,
                severity="critical"
            ) 

        is_retry = any(word in content_lower for word in RETRY_WORDS)
        if is_retry:
            self.retry_count += 1
            if self.retry_count >= 2:
                self._log_flag(
                    "RETRY_LOOP",
                    "Agent appears to be retrying repeatedly with no stopping condition",
                    evidence=agent_text,
                    severity="medium"
                )
        else:
            self.retry_count = 0

        if agent_text.lower().startswith("warning:"):
            self._log_flag(
                "RISK_FLAG",
                "Step reported warning status with risk flags",
                evidence=agent_text,
                severity="medium"
            )

        if agent_text.lower().startswith("critical:") or agent_text.lower().startswith("system error:"):
            self._log_flag(
                "CRITICAL_SYSTEM_FAILURE",
                "System reported a critical error",
                evidence=agent_text,
                severity="critical"
            )

        if any(word in content_lower for word in BOOKING_CLAIMS) and not self.tool_was_called:
            self._log_flag(
                "BOOKING_CLAIM_WITHOUT_TOOL",
                "Agent claimed a booking/purchase/reservation without any tool call",
                evidence=agent_text,
                severity="critical"
            )

        curr_numbers = extract_numbers(agent_text)
        for prev_step, prev_claim in self.agent_claims:
            prev_numbers = extract_numbers(prev_claim)
            for pn in prev_numbers:
                pval = float(pn)
                if abs(pval) < 10:
                    continue
                for cn in curr_numbers:
                    cval = float(cn)
                    if abs(cval) > 10 and abs(cval - pval) > 0.01:
                        prev_words = set(prev_claim.lower().split())
                        curr_words = set(agent_text.lower().split())
                        common_words = {"the", "a", "an", "is", "are", "was", "and", "or", "for", "to", "in", "of", "your", "i", "it"}
                        shared = (prev_words & curr_words) - common_words
                        if len(shared) >= 2:
                            self._log_flag(
                                "CONTEXT_DROP",
                                f"Agent contradicted its own earlier statement (previously stated {pn}, now states {cn})",
                                evidence=agent_text,
                                severity="critical"
                            )

        layer1_found_issue = len(self.flags) > flags_before_this_check

        if self.use_semantic_layer and not layer1_found_issue:
            context = self.last_tool_output or ""
            if semantic_check(agent_text, context):
                self._log_flag(
                    "SEMANTIC_ANOMALY",
                    "GPT-4o-mini flagged a subtle issue not caught by deterministic checks",
                    evidence=agent_text,
                    severity="medium"
                )

        self.agent_claims.append((len(self.steps), agent_text))

        for mechanism, claim_phrases in MECHANISM_CLAIMS.items():
            claims_mechanism = any(phrase in content_lower for phrase in claim_phrases)
            if not claims_mechanism:
                continue
            observable_indicators = OBSERVABLE_INDICATORS[mechanism]
            evidence_found = any(
                any(ind in past_content for ind in observable_indicators)
                for past_content in self.all_content_history[:-1]
            )
            if not evidence_found:
                self._log_flag(
                    "UNVERIFIABLE_ASSERTION",
                    f"Agent asserted {mechanism.replace('_', ' ')} executed but no observable evidence exists in the trace",
                    evidence=agent_text,
                    severity="high"
                )

        if is_validator_complaint:
            topics = extract_topic_keywords(agent_text)
            self.validator_topics.append(topics)

            if len(self.validator_topics) >= 2:
                for i in range(len(self.validator_topics) - 1):
                    overlap = self.validator_topics[i] & topics
                    if len(overlap) >= 2:
                        self._log_flag(
                            "THEMATIC_OSCILLATION",
                            f"Current complaint shares core issue with round {i + 1}",
                            evidence=f"Shared concepts: {overlap}"
                        )
                        break

    def summary(self) -> dict:
        return {
            "total_steps": len(self.steps),
            "total_flags": len(self.flags),
            "flags": self.flags,
        }

    def should_block(self) -> bool:
        was_blocked = self.blocked
        self.blocked = False
        return was_blocked

