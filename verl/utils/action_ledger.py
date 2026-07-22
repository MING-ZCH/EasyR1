"""Strict token-native action-event ledger helpers."""

import hashlib
import json
from numbers import Integral
from typing import Any, Iterable, Sequence


ACTION_TYPE_IDS = {"point": 1, "answer": 2, "cap": 3, "abort": 4}
ACTION_LEDGER_VERSION = 2
ACTION_TAG_TOKEN_IDS = {
    "answer_open": (151667,),
    "answer_close": (151668,),
    "point_open": (151669,),
    "point_close": (151670,),
}
_EVENT_FIELDS = (
    "ordinal",
    "type",
    "span_start",
    "span_end",
    "decision_start",
    "decision_end",
    "closed",
    "truncated",
    "turn_index",
    "stop_reason",
)
_CAP_REASONS = {"token_cap", "turn_cap"}
_ABORT_REASONS = {
    "eos",
    "immediate_eos",
    "empty_completion",
    "generation_error",
    "abort",
    "invalid_native_span",
    "stop_semantics_violation",
}


def classify_empty_generation(
    *, completion_present: bool, stop_reason: Any = None, finish_reason: Any = None
) -> str:
    """Classify a zero-token scheduler result without conflating it with a cap."""
    if not completion_present:
        return "generation_error"
    terminal_reason = stop_reason if stop_reason is not None else finish_reason
    if str(terminal_reason).strip().lower() in ("eos", "stop", "stopped"):
        return "immediate_eos"
    return "empty_completion"


def response_token_digest(token_ids: Sequence[int]) -> str:
    """Return a stable digest over the exact ordered response token IDs."""
    normalized = [_strict_int("response token id", token_id) for token_id in token_ids]
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def action_event_ledger_digest(events: Sequence[dict[str, Any]]) -> str:
    """Return a deterministic digest over every event field and event order."""
    if not isinstance(events, (list, tuple)):
        raise ValueError("action ledger events must be a list or tuple.")
    try:
        payload = json.dumps(
            list(events),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("action ledger events must have a canonical JSON representation.") from exc
    return hashlib.sha256(payload).hexdigest()


def canonical_interleaved_stop_sequences() -> tuple[str, str]:
    """The V37 scheduler stops on the first complete action-closing tag."""
    return ("</point>", "</answer>")


def validate_action_tag_token_ids(tag_token_ids: Any, *, require_expected: bool = True) -> dict[str, tuple[int, ...]]:
    """Validate the tokenizer-side control-tag identity contract."""
    if not isinstance(tag_token_ids, dict) or set(tag_token_ids) != set(ACTION_TAG_TOKEN_IDS):
        raise ValueError("action tag token contract must contain exactly the four control tags.")
    normalized: dict[str, tuple[int, ...]] = {}
    for name, expected in ACTION_TAG_TOKEN_IDS.items():
        value = tag_token_ids[name]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must have a nonempty token-id sequence.")
        ids = tuple(_strict_int(f"{name} token id", token_id) for token_id in value)
        if require_expected and ids != expected:
            raise ValueError(f"{name} must encode as the singleton {expected}, got {ids}.")
        normalized[name] = ids
    return normalized


def make_action_ledger_envelope(
    events: Sequence[dict[str, Any]], token_ids: Sequence[int], tag_token_ids: dict[str, Sequence[int]]
) -> dict[str, Any]:
    """Bind a ledger to its schema, tokenizer tag IDs, and exact response IDs."""
    tags = validate_action_tag_token_ids(tag_token_ids, require_expected=False)
    return {
        "version": ACTION_LEDGER_VERSION,
        "response_token_digest": response_token_digest(token_ids),
        "event_ledger_digest": action_event_ledger_digest(events),
        "tag_token_ids": {name: list(ids) for name, ids in tags.items()},
        "events": [dict(event) for event in events],
    }


def make_action_event(
    *,
    ordinal: int,
    event_type: str,
    span_start: int,
    span_end: int,
    decision_start: int,
    decision_end: int,
    closed: bool,
    truncated: bool,
    turn_index: int,
    stop_reason: Any,
) -> dict[str, Any]:
    """Construct an event without coercing untrusted values.

    Callers produce native Python scalars already.  Keeping this constructor
    strict prevents values such as ``1.5`` or ``"false"`` from being silently
    converted into a valid-looking ledger.
    """
    return {
        "ordinal": ordinal,
        "type": event_type,
        "span_start": span_start,
        "span_end": span_end,
        "decision_start": decision_start,
        "decision_end": decision_end,
        "closed": closed,
        "truncated": truncated,
        "turn_index": turn_index,
        "stop_reason": stop_reason,
    }


def _strict_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _strict_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return value


def _control_tag_occurrences(
    tokens: Sequence[int], tag_token_ids: dict[str, tuple[int, ...]]
) -> list[tuple[int, str]]:
    occurrences: list[tuple[int, str]] = []
    for position in range(len(tokens)):
        names = [
            name
            for name, pattern in tag_token_ids.items()
            if list(tokens[position : position + len(pattern)]) == list(pattern)
        ]
        if len(names) > 1:
            raise ValueError("action control-tag token sequences overlap ambiguously.")
        if names:
            occurrences.append((position, names[0]))
    return occurrences


def _canonical_native_action_events(
    token_ids: Sequence[int], tag_token_ids: dict[str, tuple[int, ...]]
) -> tuple[list[dict[str, Any]], bool]:
    """Reconstruct the unique closed-action sequence from native response IDs."""
    tokens = list(token_ids)
    events: list[dict[str, Any]] = []
    occurrences = _control_tag_occurrences(tokens, tag_token_ids)
    occurrence_index = 0
    answer_seen = False
    while occurrence_index < len(occurrences):
        start, tag_name = occurrences[occurrence_index]
        if answer_seen or not tag_name.endswith("_open"):
            return [], True
        event_type = tag_name.removesuffix("_open")
        close_name = f"{event_type}_close"
        if occurrence_index + 1 >= len(occurrences):
            return [], True
        close_start, observed_close_name = occurrences[occurrence_index + 1]
        if observed_close_name != close_name:
            return [], True
        end = close_start + len(tag_token_ids[close_name])
        ordinal = len(events)
        events.append(
            make_action_event(
                ordinal=ordinal,
                event_type=event_type,
                span_start=start,
                span_end=end,
                decision_start=start,
                decision_end=end,
                closed=True,
                truncated=False,
                turn_index=ordinal,
                stop_reason=event_type,
            )
        )
        answer_seen = event_type == "answer"
        occurrence_index += 2
        if answer_seen and (end != len(tokens) or occurrence_index != len(occurrences)):
            return [], True
    return events, False


def _invalid_native_abort(response_length: int) -> dict[str, Any]:
    return make_action_event(
        ordinal=0,
        event_type="abort",
        span_start=response_length,
        span_end=response_length,
        decision_start=response_length,
        decision_end=response_length,
        closed=False,
        truncated=False,
        turn_index=0,
        stop_reason="invalid_native_span",
    )


def validate_action_event_row(
    events: Any,
    response_length: int,
    *,
    response_token_ids: Sequence[int] | None = None,
    require_token_binding: bool = False,
) -> list[dict[str, Any]]:
    """Validate one complete ledger row.

    Point and answer events must be closed, positive-length native spans.  Cap
    and abort events are terminal, uncredited boundary markers and therefore
    must be zero-length at ``response_length``.  All credited spans are ordered
    and pairwise disjoint.
    """
    response_length = _strict_int("response_length", response_length)
    if response_length < 0:
        raise ValueError(f"response_length must be nonnegative, got {response_length}.")
    tag_token_ids = None
    if isinstance(events, dict):
        expected_keys = {
            "version",
            "response_token_digest",
            "event_ledger_digest",
            "tag_token_ids",
            "events",
        }
        if set(events) != expected_keys:
            raise ValueError("action ledger envelope has missing or unknown fields.")
        version = _strict_int("action ledger version", events.get("version"))
        if version != ACTION_LEDGER_VERSION:
            raise ValueError(f"unsupported action ledger version: {version!r}.")
        tag_token_ids = validate_action_tag_token_ids(
            events.get("tag_token_ids"), require_expected=require_token_binding
        )
        if response_token_ids is None:
            if require_token_binding:
                raise ValueError("token-bound action ledger validation requires response token IDs.")
        elif events.get("response_token_digest") != response_token_digest(response_token_ids):
            raise ValueError("action ledger response-token digest mismatch.")
        envelope_events = events.get("events")
        if events.get("event_ledger_digest") != action_event_ledger_digest(envelope_events):
            raise ValueError("action ledger event-ledger digest mismatch.")
        events = envelope_events
    elif require_token_binding:
        raise ValueError("V37 action mode requires a versioned token-bound action ledger envelope.")
    if not isinstance(events, (list, tuple)):
        raise ValueError("action ledger events must be a list or tuple.")
    if not events:
        raise ValueError("action ledger row must contain at least one event.")
    if response_token_ids is not None and len(response_token_ids) != response_length:
        raise ValueError("response token IDs must exactly match response_length.")

    normalized: list[dict[str, Any]] = []
    previous_end = 0
    answer_seen = False
    terminal_marker_seen = False
    for expected_ordinal, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("each action ledger event must be a mapping.")
        if set(event) != set(_EVENT_FIELDS):
            raise ValueError("action ledger event has missing or unknown fields.")
        event_type = event.get("type")
        if event_type not in ACTION_TYPE_IDS:
            raise ValueError(f"unsupported action event type: {event_type!r}.")
        ordinal = _strict_int("ordinal", event.get("ordinal"))
        if ordinal != expected_ordinal:
            raise ValueError(f"action ordinals must be contiguous: {ordinal} != {expected_ordinal}.")
        span_start = _strict_int("span_start", event.get("span_start"))
        span_end = _strict_int("span_end", event.get("span_end"))
        decision_start = _strict_int("decision_start", event.get("decision_start"))
        decision_end = _strict_int("decision_end", event.get("decision_end"))
        closed = _strict_bool("closed", event.get("closed"))
        truncated = _strict_bool("truncated", event.get("truncated"))
        turn_index = _strict_int("turn_index", event.get("turn_index"))
        if turn_index < 0:
            raise ValueError(f"turn_index must be nonnegative, got {turn_index}.")
        stop_reason = event.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ValueError(f"stop_reason must be a string or null, got {stop_reason!r}.")

        is_terminal_marker = event_type in ("cap", "abort")
        if is_terminal_marker:
            if answer_seen:
                raise ValueError("a terminal answer cannot be followed by a cap or abort marker.")
            if ordinal != len(events) - 1:
                raise ValueError(f"{event_type} must be the final ledger event.")
            if (span_start, span_end, decision_start, decision_end) != (
                response_length,
                response_length,
                response_length,
                response_length,
            ):
                raise ValueError(f"{event_type} must be an uncredited marker at the response boundary.")
            if closed:
                raise ValueError(f"{event_type} marker cannot be closed.")
            if turn_index != expected_ordinal:
                raise ValueError(
                    f"{event_type} marker turn_index must equal its canonical action boundary {expected_ordinal}."
                )
            if event_type == "cap":
                if not truncated or stop_reason not in _CAP_REASONS:
                    raise ValueError("cap requires truncated=true and a token_cap/turn_cap reason.")
            elif truncated or stop_reason not in _ABORT_REASONS:
                raise ValueError("abort requires truncated=false and an explicit abort reason.")
            terminal_marker_seen = True
        else:
            if terminal_marker_seen or answer_seen:
                raise ValueError("legal action order is zero or more points followed by at most one answer.")
            if not closed or truncated:
                raise ValueError(f"{event_type} events must be closed and non-truncated.")
            if not (0 <= span_start < span_end <= response_length):
                raise ValueError(
                    f"invalid event span [{span_start}, {span_end}) for response length {response_length}."
                )
            if (decision_start, decision_end) != (span_start, span_end):
                raise ValueError(
                    f"{event_type} decision span must exactly equal its full closed event span."
                )
            if turn_index != expected_ordinal:
                raise ValueError(
                    f"{event_type} turn_index must equal canonical closed-action index {expected_ordinal}."
                )
            if stop_reason != event_type:
                raise ValueError(f"{event_type} stop_reason must be exactly {event_type!r}.")
            if event_type == "answer" and span_end != response_length:
                raise ValueError("the terminal answer must end at the response boundary.")
            if span_start < previous_end:
                raise ValueError(f"overlapping or out-of-order action span starts at token {span_start}.")
            if response_token_ids is not None and tag_token_ids is not None:
                open_ids = tag_token_ids[f"{event_type}_open"]
                close_ids = tag_token_ids[f"{event_type}_close"]
                actual = list(response_token_ids[span_start:span_end])
                if actual[: len(open_ids)] != list(open_ids) or actual[-len(close_ids) :] != list(close_ids):
                    raise ValueError(
                        f"{event_type} span [{span_start}, {span_end}) does not match response tag token IDs."
                    )
            previous_end = span_end
            answer_seen = event_type == "answer"

        normalized.append(
            {
                "ordinal": ordinal,
                "type": event_type,
                "span_start": span_start,
                "span_end": span_end,
                "decision_start": decision_start,
                "decision_end": decision_end,
                "closed": closed,
                "truncated": truncated,
                "turn_index": turn_index,
                "stop_reason": stop_reason,
            }
        )
    if response_token_ids is not None and tag_token_ids is not None:
        reconstructed, native_invalid = _canonical_native_action_events(response_token_ids, tag_token_ids)
        stop_semantics_abort = (
            len(normalized) == 1
            and normalized[0]["type"] == "abort"
            and normalized[0]["stop_reason"] == "stop_semantics_violation"
        )
        if native_invalid:
            expected = [_invalid_native_abort(response_length)]
            if normalized != expected:
                raise ValueError("malformed native action tags require the canonical invalid_native_span abort.")
        elif not stop_semantics_abort:
            ledger_actions = [event for event in normalized if event["type"] in ("point", "answer")]
            if ledger_actions != reconstructed:
                raise ValueError("action ledger events do not equal the canonical native-token reconstruction.")
            if reconstructed and reconstructed[-1]["type"] == "answer":
                if normalized != reconstructed:
                    raise ValueError("the closed answer must be the terminal ledger event.")
            elif require_token_binding:
                markers = [event for event in normalized if event["type"] in ("cap", "abort")]
                if len(markers) != 1 or normalized[:-1] != reconstructed:
                    raise ValueError("a token-bound row without an answer requires one canonical terminal marker.")
    return normalized


def ledger_to_dense(
    events: Iterable[dict[str, Any]], values: Iterable[float], response_length: int
) -> tuple[list[float], list[float], list[int]]:
    """Convert a validated row to token-dense channels."""
    events = validate_action_event_row(list(events), response_length)
    values = [float(value) for value in values]
    if len(events) != len(values):
        raise ValueError(f"action event/value count mismatch: {len(events)} != {len(values)}.")
    mask = [0.0] * response_length
    dense_values = [0.0] * response_length
    types = [0] * response_length
    for event, value in zip(events, values):
        if event["type"] in ("cap", "abort"):
            if value != 0.0:
                raise ValueError(f"zero-length {event['type']} marker cannot carry local action credit.")
            continue
        for pos in range(event["decision_start"], event["decision_end"]):
            if mask[pos] != 0:
                raise ValueError(f"overlapping action decision spans at token {pos}.")
            mask[pos] = 1.0
            dense_values[pos] = value
            types[pos] = ACTION_TYPE_IDS[event["type"]]
    return mask, dense_values, types


def find_token_subsequence(sequence: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    """Return the first exact native-token match, or ``-1``."""
    if not pattern:
        return -1
    stop = len(sequence) - len(pattern) + 1
    for idx in range(max(0, start), max(0, stop)):
        if list(sequence[idx : idx + len(pattern)]) == list(pattern):
            return idx
    return -1


def build_native_action_event_row(
    token_ids: Sequence[int],
    *,
    point_open_ids: Sequence[int],
    point_close_ids: Sequence[int],
    answer_open_ids: Sequence[int],
    answer_close_ids: Sequence[int],
    turn_spans: Sequence[tuple[int, int, int]],
    termination_type: str | None,
    termination_reason: str | None,
    return_envelope: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Build exact spans solely from generated token IDs.

    Any unmatched control tag or non-native tag mapping fails closed to one
    abort marker.  This deliberately does not decode or re-tokenize reward text.
    """
    tokens = [_strict_int("response token id", token_id) for token_id in token_ids]
    tag_token_ids = validate_action_tag_token_ids(
        {
            "point_open": point_open_ids,
            "point_close": point_close_ids,
            "answer_open": answer_open_ids,
            "answer_close": answer_close_ids,
        },
        require_expected=False,
    )
    events, native_invalid = _canonical_native_action_events(tokens, tag_token_ids)
    stop_semantics_invalid = False

    # With the production singleton tags, every closed action must end the
    # generation request that produced it, and no request may close two
    # actions.  Public turn_index remains the gap-free action-turn ordinal so
    # reward ingress can reconstruct it from response IDs alone.
    if not native_invalid and tag_token_ids == ACTION_TAG_TOKEN_IDS and tokens:
        normalized_turn_spans: list[tuple[int, int, int]] = []
        previous_end = 0
        previous_turn = -1
        try:
            for raw_span in turn_spans:
                if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 3:
                    raise ValueError("native turn span must be a (start, end, turn_index) triple.")
                start = _strict_int("native turn start", raw_span[0])
                end = _strict_int("native turn end", raw_span[1])
                turn_index = _strict_int("native turn index", raw_span[2])
                if start != previous_end or not (start < end <= len(tokens)) or turn_index <= previous_turn:
                    raise ValueError("native turn spans must be contiguous, nonempty, and strictly ordered.")
                normalized_turn_spans.append((start, end, turn_index))
                previous_end = end
                previous_turn = turn_index
            if previous_end != len(tokens):
                raise ValueError("native turn spans must cover the exact response token sequence.")
            closed_in_turn: set[int] = set()
            closing_span_index = 0
            for event in events:
                closing_position = event["span_end"] - 1
                while (
                    closing_span_index < len(normalized_turn_spans)
                    and normalized_turn_spans[closing_span_index][1] <= closing_position
                ):
                    closing_span_index += 1
                closing_span = (
                    normalized_turn_spans[closing_span_index]
                    if closing_span_index < len(normalized_turn_spans)
                    else None
                )
                if closing_span is None or event["span_end"] != closing_span[1] or closing_span[2] in closed_in_turn:
                    raise ValueError("each native generation turn must end with at most one closed action.")
                closed_in_turn.add(closing_span[2])
        except (TypeError, ValueError):
            stop_semantics_invalid = True

    if native_invalid or (events and events[-1]["type"] == "answer" and termination_type is not None):
        events = [_invalid_native_abort(len(tokens))]
    elif stop_semantics_invalid:
        events = [
            make_action_event(
                ordinal=0,
                event_type="abort",
                span_start=len(tokens),
                span_end=len(tokens),
                decision_start=len(tokens),
                decision_end=len(tokens),
                closed=False,
                truncated=False,
                turn_index=0,
                stop_reason="stop_semantics_violation",
            )
        ]
    elif termination_type in ("cap", "abort") or (events and events[-1]["type"] == "point"):
        marker_type = termination_type or "abort"
        marker_reason = termination_reason or "abort"
        events.append(
            make_action_event(
                ordinal=len(events),
                event_type=marker_type,
                span_start=len(tokens),
                span_end=len(tokens),
                decision_start=len(tokens),
                decision_end=len(tokens),
                closed=False,
                truncated=marker_type == "cap",
                turn_index=len(events),
                stop_reason=marker_reason,
            )
        )
    elif termination_type is not None:
        raise ValueError(f"unsupported termination type: {termination_type!r}.")
    if not events:
        events.append(
            make_action_event(
                ordinal=0,
                event_type="abort",
                span_start=len(tokens),
                span_end=len(tokens),
                decision_start=len(tokens),
                decision_end=len(tokens),
                closed=False,
                truncated=False,
                turn_index=0,
                stop_reason="abort",
            )
        )
    envelope = make_action_ledger_envelope(events, tokens, tag_token_ids)
    validated = validate_action_event_row(
        envelope, len(tokens), response_token_ids=tokens, require_token_binding=False
    )
    return envelope if return_envelope else validated
