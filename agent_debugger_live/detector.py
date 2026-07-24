import re
import os
from langchain_core.callbacks import BaseCallbackHandler
from openai import OpenAI

SUCCESS_CLAIMS = [
    "successfully", "confirmed", "completed", "trading at",
    "processed", "has been delivered", "was delivered", "delivered successfully",
    "is ready", "is confirmed", "is complete", "is being processed",
    "will process", "will refund", "right away", "i will process"
]

NEGATION_WORDS = ["not", "cannot", "unable", "couldn't", "wasn't", "isn't", "doesn't", "no", "does not", "did not", "unknown", "unavailable"]

# Distinct from negation: these mark a claim as a FUTURE promise, not a present
# success statement - "we'll share it once we have confirmed information" is
# not the same claim as "it is confirmed". Caught separately because no
# negation word widening would ever find these; the issue is tense, not distance.
FUTURE_CONDITIONAL_MARKERS = ["as soon as we", "once we have", "once we receive",
                              "when we have", "will share", "will provide",
                              "will update you once", "as soon as it is"]


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
    # (?<![A-Za-z0-9]) prevents the hyphen in IDs like "ORD-4471"
    # from being read as a negative sign, while real negatives ("-42") still match.
    return [n.rstrip('.') for n in re.findall(r'(?<![A-Za-z0-9])-?\d+\.?\d*', text)]


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


def check_success_claim(content_lower: str) -> bool:
    """
    Checks if text contains a genuine success claim, ignoring matches that
    are negated ANYWHERE in the same sentence (e.g. 'not found', 'no shipping
    carrier or tracking number can be confirmed', 'cannot currently be
    confirmed'). Scoped to the sentence containing the claim word - not a
    fixed character count - because negation can sit an arbitrary distance
    before the claim word depending on how long the clause in between is.
    A fixed-width window was tried first and kept missing longer clauses;
    sentence-scoping fixes the whole failure class at once instead of one
    phrasing at a time.
    """
    sentences = re.split(r'(?<=[.!?])\s+', content_lower)
    for word in SUCCESS_CLAIMS:
        idx = content_lower.find(word)
        if idx == -1:
            continue
        prefix_negated = content_lower[max(0, idx - 2):idx] == "un"  # e.g. "unprocessed"
        if prefix_negated:
            continue
        running = 0
        sentence = content_lower
        for s in sentences:
            if running <= idx < running + len(s) + 1:
                sentence = s
                break
            running += len(s) + 1
        if not any(neg in sentence for neg in NEGATION_WORDS) and \
           not any(marker in sentence for marker in FUTURE_CONDITIONAL_MARKERS):
            return True
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
        # Cross-node propagation tracking:
        # when a node's response gets flagged high/critical, its numbers and
        # keywords become "tainted". If a DIFFERENT node later repeats them,
        # the failure has propagated across the graph.
        self.tainted_claims = []  # list of dicts: {node, numbers, keywords, flag_type}

    def register_context(self, text: str):
        """Feed the handler known context (the user's request, order IDs, amounts)
        so those values are never mistaken for numbers an agent invented.
        Call once at the start of a session/pipeline run."""
        self.all_content_history.append(str(text).lower())

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
        if hasattr(output, "content"):
            content = str(output.content)
        else:
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

    def check_agent_response(self, agent_text: str, is_validator_complaint: bool = False, node: str = None):
        content_lower = agent_text.lower()
        claims_success = check_success_claim(content_lower)
        flags_before = len(self.flags)

        # --- FABRICATED_IDENTIFIER: agent admits the lookup failed, but still
        # states a specific tracking/order/reference number as if it were real.
        # Different from HALLUCINATION (which needs a success-claim word like
        # "processed") - this catches a lie hiding INSIDE an honest admission,
        # e.g. "untraceable in our system, use tracking number 1Z2345W67890".
        ADMISSION_PHRASES = ["untraceable", "not found", "no record", "no matching",
                             "cannot be found", "could not be found", "no matching record"]
        # A plausible invented identifier: a token mixing letters+digits (like a
        # real tracking number) that's long enough to look specific, and that
        # never appeared in the tool's actual output or prior real context.
        ID_PATTERN = re.compile(r'\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{8,}\b', re.IGNORECASE)
        if self.last_tool_output is not None and not self.last_tool_output.strip():
            if any(p in content_lower for p in ADMISSION_PHRASES):
                known_context = " ".join(self.all_content_history).lower()
                for candidate in ID_PATTERN.findall(agent_text):
                    if candidate.lower() not in known_context:
                        self._log_flag(
                            "FABRICATED_IDENTIFIER",
                            f"Agent admitted the lookup failed, but stated a specific "
                            f"identifier ('{candidate}') that never came from the tool "
                            f"or prior context - likely invented",
                            evidence=agent_text,
                            severity="critical"
                        )
                        break

        # --- CROSS_NODE_PROPAGATION check (runs first, against prior tainted claims) ---
        if node is not None:
            curr_nums = {n for n in extract_numbers(agent_text) if abs(float(n)) > 10}
            curr_keys = extract_topic_keywords(agent_text)
            for taint in self.tainted_claims:
                if taint["node"] == node:
                    continue  # same node repeating itself is not propagation
                shared_nums = curr_nums & taint["numbers"]
                shared_keys = curr_keys & taint["keywords"]
                if shared_nums or len(shared_keys) >= 3:
                    detail = (
                        f"repeated unverified value(s) {sorted(shared_nums)}"
                        if shared_nums
                        else f"repeated flagged concepts {sorted(shared_keys)[:5]}"
                    )
                    self._log_flag(
                        "CROSS_NODE_PROPAGATION",
                        f"Node '{node}' consumed a flagged claim from node '{taint['node']}' "
                        f"({taint['flag_type']}) and {detail} as if verified",
                        evidence=agent_text,
                        severity="critical"
                    )
                    break

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
                if pn in curr_numbers:
                    continue  # agent restates the same value consistently - not a contradiction
                for cn in curr_numbers:
                    cval = float(cn)
                    if abs(cval - pval) > pval * 0.5:
                        continue  # different magnitude = different quantity (order ID vs tracking number), not a contradiction
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

        flags_before_this_check = len(self.flags)
        layer1_found_issue = False

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

        # --- Taint registration: if THIS response was flagged high/critical,
        # remember its numbers/keywords so downstream nodes repeating them get caught ---
        if node is not None:
            new_flags = self.flags[flags_before:]
            serious = [f for f in new_flags
                       if f["severity"] in ("high", "critical")
                       and f["type"] != "CROSS_NODE_PROPAGATION"]
            if serious:
                # Only taint numbers this node INTRODUCED. Numbers already present
                # earlier in the session (order IDs, amounts from the user's request)
                # are shared context, not evidence of propagation - every honest
                # downstream node will legitimately repeat them.
                prior_text = " ".join(self.all_content_history[:-1])
                introduced = {n for n in extract_numbers(agent_text)
                              if abs(float(n)) > 10 and n not in prior_text}
                self.tainted_claims.append({
                    "node": node,
                    "numbers": introduced,
                    "keywords": extract_topic_keywords(agent_text),
                    "flag_type": serious[0]["type"],
                })

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

    def get_safe_response(self, original_response: str) -> str:
        if self.should_block():
            return "[This response was withheld due to a detected reliability issue. A human has been notified.]"
        return original_response
