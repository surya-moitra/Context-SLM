"""Source-grounded temporal assertions, independent of question wording.

The grammar operates on evidence spans, not benchmark questions. Dates and
offsets are computed here; a model is never asked to invent normalized dates.
"""

import calendar
import datetime
import re


MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
CARDINALS = dict(zip(
    ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen").split(), range(20),
))
CARDINALS.update(dict(zip(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10),
)))
ORDINALS = dict(zip(
    ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
     "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
     "nineteenth").split(), range(1, 20),
))
ORDINALS.update(dict(zip(
    "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split(),
    range(20, 100, 10),
)))
UNITS = {"day", "week", "month", "year"}


def words(text):
    return [
        {"word": m.group().lower(), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"[A-Za-z]+|\d+(?:\.\d+)?", str(text or ""))
    ]


def calendar_date(value):
    match = re.search(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", str(value or ""))
    if match:
        try:
            return datetime.date(*(int(p) for p in match.groups()))
        except ValueError:
            pass
    return None


def number_at(tokens, start):
    """Consume a complete cardinal/ordinal phrase, including cardinal 'and'."""
    total, current, end = 0, 0, start
    for index in range(start, len(tokens)):
        word = tokens[index]["word"]
        if word == "and" and end > start:
            if index + 1 < len(tokens) and tokens[index + 1]["word"] in CARDINALS | ORDINALS:
                continue
            break
        if word in {"a", "an"} and index == start:
            current += 1
        elif word.replace(".", "", 1).isdigit():
            if end > start:
                break
            current = float(word)
            if index + 1 == len(tokens) or tokens[index + 1]["word"] not in {"hundred", "thousand"}:
                return current, index + 1
        elif word in {"hundred", "hundredth"}:
            current = max(1, current) * 100
        elif word in {"thousand", "thousandth"}:
            total += max(1, current) * 1000
            current = 0
        elif word in CARDINALS | ORDINALS:
            current += (CARDINALS | ORDINALS)[word]
        else:
            break
        end = index + 1
        if word in ORDINALS or word in {"hundredth", "thousandth"}:
            break
    return (total + current, end) if end > start else (None, start)


def shift_date(value, amount, unit):
    if unit == "day":
        return value + datetime.timedelta(days=amount)
    if unit == "week":
        return value + datetime.timedelta(weeks=amount)
    months = amount * (12 if unit == "year" else 1)
    index = value.year * 12 + value.month - 1 + months
    year, month = divmod(index, 12)
    return datetime.date(year, month + 1, min(value.day, calendar.monthrange(year, month + 1)[1]))


def point(value, basis, source_timestamp, span=None):
    return {
        "value": value.isoformat(), "granularity": "day",
        "interval_start": value.isoformat(), "interval_end": value.isoformat(),
        "basis": basis, "source_timestamp": source_timestamp,
        "time_role": "event", "time_span": span,
        "display": f"{value.day} {value.strftime('%B %Y')}",
    }


def evidence_phase(text, default="completed"):
    tokens = [t["word"] for t in words(text)]
    planning = {"plan", "plans", "planned", "planning", "considered", "considering",
                "intend", "intended", "tentative", "scheduled", "will"}
    irregular = {"bought", "met", "born", "took", "went", "ran", "got", "made", "did"}
    reporting = {"recorded", "reported", "mentioned", "noted", "said", "wrote", "confirmed", "discussed", "explained"}
    for index, token in enumerate(tokens):
        if token in reporting:
            continue
        if token in planning:
            return "planned"
        if token.endswith("ed") or token in irregular:
            if (any(t in {"never", "not"} for t in tokens[max(0, index - 3):index])
                    or (token == "did" and index + 1 < len(tokens) and tokens[index + 1] == "not")):
                return "negated_or_cancelled"
            return "completed"
    return default


def split_assertions(text):
    """Separate coordinated assertions only at an explicit new subject."""
    tokens = words(text)
    boundaries = [0]
    subjects = {"i", "we", "he", "she", "they", "you"}
    for i, token in enumerate(tokens[:-1]):
        if token["word"] in {"but", "and", "then"} and tokens[i + 1]["word"] in subjects:
            boundaries.append(tokens[i + 1]["start"])
    boundaries.append(len(text))
    return [text[a:b].strip(" ,;") for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip(" ,;")]


def describe_time(text, source_timestamp, existing=None, default_status="completed"):
    """Describe one source assertion and its grounded point/interval/relation."""
    tokens = words(text)
    plain = [t["word"] for t in tokens]
    source = calendar_date(source_timestamp)
    expression = existing
    relation = None
    invalid_explicit = False
    event_end = len(text)
    explicit = bool(existing) or bool(re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text))
    explicit = explicit or any(w in {
        "ago", "earlier", "later", "previously", "before", "after",
        "today", "yesterday", "tomorrow",
    } for w in plain)
    explicit = explicit or any(
        (w.rstrip("s") in UNITS and any(p in {"last", "next", "this", "previous"}
                                       for p in plain[max(0, i - 2):i]))
        or (w in WEEKDAYS | MONTHS and any(p in {"on", "in", "during", "since", "of"}
                                         for p in plain[max(0, i - 2):i]))
        for i, w in enumerate(plain)
    )
    numeric_phrases = []
    index = 0
    while index < len(tokens):
        value, end = number_at(tokens, index)
        if value is None:
            index += 1
            continue
        numeric_phrases.append((index, end, value))
        index = end

    for start, end, value in numeric_phrases:
        unit_index = end + int(end < len(plain) and plain[end] == "calendar")
        if unit_index >= len(plain) or plain[unit_index].rstrip("s") not in UNITS:
            continue
        unit = plain[unit_index].rstrip("s")
        after = unit_index + 1
        if after >= len(plain) or value != int(value):
            continue
        amount = int(value)
        if plain[after] == "ago" and source:
            event_end = min(event_end, tokens[start]["start"])
            if unit == "day":
                expression = point(shift_date(source, -amount, unit), "source_relative_offset", source_timestamp,
                                   [tokens[start]["start"], tokens[after]["end"]])
            elif unit == "week" and any(w in WEEKDAYS for w in plain[:start]):
                weekday = next(WEEKDAYS[w] for w in reversed(plain[:start]) if w in WEEKDAYS)
                date = source - datetime.timedelta(days=source.weekday(), weeks=amount)
                expression = point(date + datetime.timedelta(days=weekday), "source_relative_weekday", source_timestamp)
        elif plain[after] in {"before", "after"}:
            reference = text[tokens[after]["end"]:].strip(" ,.;!?")
            reference_words = {t["word"] for t in words(reference)}
            report_reference = reference_words.issubset({"this", "that", "the", "message", "report", "session", "conversation"})
            if not report_reference and reference:
                relation = {"amount": amount, "unit": unit, "direction": plain[after],
                            "reference_text": reference,
                            "span": [tokens[start]["start"], len(text)]}
                expression = None
                event_end = min(event_end, tokens[start]["start"])

    if source and any(w in WEEKDAYS for w in plain):
        weekday = next(WEEKDAYS[w] for w in plain if w in WEEKDAYS)
        for i, w in enumerate(plain):
            if w == "week" and any(p in {"last", "previous"} for p in plain[max(0, i - 2):i]):
                date = source - datetime.timedelta(days=source.weekday(), weeks=1)
                expression = point(date + datetime.timedelta(days=weekday), "source_relative_weekday", source_timestamp)

    for start, end, value in numeric_phrases:
        if not 1 <= value <= 31 or value != int(value):
            continue
        cursor = end
        if cursor < len(plain) and plain[cursor] in {"st", "nd", "rd", "th"}:
            cursor += 1
        if cursor < len(plain) and plain[cursor] == "of":
            cursor += 1
        if cursor >= len(plain):
            continue
        month = None
        year = source.year if source else None
        if plain[cursor] in MONTHS:
            month = MONTHS[plain[cursor]]
            if cursor + 1 < len(plain) and plain[cursor + 1].isdigit() and len(plain[cursor + 1]) == 4:
                year = int(plain[cursor + 1])
        elif source and plain[cursor:cursor + 2] in (["last", "month"], ["this", "month"]):
            date = shift_date(source, -1 if plain[cursor] == "last" else 0, "month")
            year, month = date.year, date.month
        elif source:
            offset, offset_end = number_at(tokens, cursor)
            tail = plain[offset_end:]
            if (offset is not None and offset == int(offset)
                    and tail[:2] == ["calendar", "months"] and "before" in tail):
                date = shift_date(source, -int(offset), "month")
                year, month = date.year, date.month
        if month and year:
            try:
                expression = point(datetime.date(year, month, int(value)), "grounded_calendar_day", source_timestamp)
                event_end = min(event_end, tokens[start]["start"])
            except ValueError:
                # An invalid explicit day must not fall back to a month/report date.
                expression = None
                explicit = True
                invalid_explicit = True
            break

    if expression is None and relation is None and not invalid_explicit:
        absolute = calendar_date(text)
        if absolute:
            expression = point(absolute, "explicit_date", source_timestamp)
        elif source:
            for word, offset in (("today", 0), ("yesterday", -1), ("tomorrow", 1)):
                if word in plain:
                    expression = point(source + datetime.timedelta(days=offset), "source_relative_" + word, source_timestamp)
                    break
    event_text = text[:event_end].strip(" ,;:.")
    status = evidence_phase(event_text, default_status)
    if text.rstrip().endswith("?"):
        status = "not_asserted"
    if expression:
        expression = {**expression, "time_role": "event", "event_status": status}
    return {"schema_version": 1, "event_text": event_text, "event_status": status,
            "explicit_time": explicit, "relation": relation, "expression": expression,
            "resolution": "resolved" if expression else "unresolved"}


def completed_calendar_months(start, end):
    start, end = sorted((start, end))
    months = (end.year - start.year) * 12 + end.month - start.month
    return months - int(shift_date(start, months, "month") > end)
