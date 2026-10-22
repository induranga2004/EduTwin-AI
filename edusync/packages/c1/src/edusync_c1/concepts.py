"""Heuristic concept extraction from lecture text and slides."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from edusync_c1.schemas import Concept

if TYPE_CHECKING:
    from edusync_c1.schemas import Page

# Common words that are not useful concepts when extracted as standalone phrases.
STOP_WORDS = frozenset(
    {
        "the", "this", "that", "these", "those", "with", "from", "into", "when",
        "where", "which", "what", "each", "such", "more", "most", "other", "some",
        "also", "should", "would", "could", "have", "been", "being", "using", "used",
        "and", "for", "are", "was", "were", "has", "had", "will", "can", "may",
        "not", "but", "all", "any", "our", "your", "their", "its", "then", "than",
        "there", "here", "they", "them", "you", "we", "it", "is", "in", "on", "at",
        "to", "of", "by", "as", "an", "a", "or", "be", "do", "does",
    }
)

# Academic, metadata, and citation words to discard from concept lists
ACADEMIC_DISCARD = frozenset(
    {
        "page", "pages", "slide", "slides", "lecture", "presentation", "course", "module",
        "chapter", "unit", "overview", "agenda", "outline", "thank you", "thanks", "questions",
        "conclusion", "objectives", "summary", "fig", "figure", "table", "et al", "bmj",
        "ieee", "acm", "csur", "tsg", "springer", "elsevier", "applied sciences", "prisma",
        "prisma 2020", "statement", "survey", "review", "paper", "report", "doi", "vol",
        "volume", "references", "bibliography", "cited", "citation", "liyanage", "author",
        "university", "department", "faculty", "school", "institute", "sliit", "malabe",
        "sri lanka", "email", "student", "supervisor", "computing", "it4041", "key references",
        "a survey", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
    }
)

# SQL statements, code commands, and syntax boilerplate to discard
SQL_DISCARD = frozenset(
    {
        "grant", "revoke", "select", "insert", "update", "delete", "create",
        "alter", "drop", "exec", "execute", "declare", "dbms_redact", "dbms_crypto",
        "sys_context", "tablespace", "redact123", "quota unlimited", "cascade",
    }
)

# Continuation headers, slide artifacts, and generic non-concept noise
GENERIC_NOISE = frozenset(
    {
        "process", "active", "data", "user", "system", "point", "case", "table",
        "file", "cpu", "type", "part", "way", "time", "level", "year", "order",
        "value", "values", "item", "items", "number", "best practices", "best practice",
        "examples cont", "examples cont.", "example", "examples", "syntax",
        "add redaction policy", "drop redaction policy", "add redaction", "drop redaction",
        "redaction policy", "step 1", "step 2",
    }
)

_PATTERNS = tuple(
    re.compile(p)
    for p in (
        # Capitalised phrases, e.g. "Binary Search Tree"
        r"\b[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*(?:\s+[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*){0,3}\b",
        # Acronyms, e.g. "TCP-IP", "DER", "SCADA"
        r"\b[A-Z]{2,}(?:[- ][A-Z0-9]{2,}){0,2}\b",
        # Lower-case multi-word phrases, e.g. "gradient descent"
        r"\b[a-zA-Z][a-zA-Z0-9-]*(?:\s+[a-zA-Z][a-zA-Z0-9-]*){1,2}\b",
    )
)


def extract_concepts(text: str, max_concepts: int = 30) -> list[Concept]:
    """Extract candidate concepts, ranked by frequency with case-insensitive dedup."""
    if not text or not text.strip() or max_concepts <= 0:
        return []

    cleaned = re.sub(r"\s+", " ", text)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    
    for pattern in _PATTERNS:
        for match in pattern.finditer(cleaned):
            phrase = match.group(0).strip(" -:•|[]()")
            words = phrase.split()
            
            # Strip leading and trailing stop words (e.g. "Cyber Resilience in" -> "Cyber Resilience")
            while words and words[0].casefold() in STOP_WORDS:
                words.pop(0)
            while words and words[-1].casefold() in STOP_WORDS:
                words.pop()
                
            if not words or len(words) > 4:
                continue
                
            clean_phrase = " ".join(words)
            if len(clean_phrase) < 3:
                continue
                
            key = clean_phrase.casefold()
            if key in ACADEMIC_DISCARD or key in GENERIC_NOISE:
                continue
            if any(w in SQL_DISCARD for w in key.split()):
                continue
            if any(d in key for d in ("liyanage", "acm csur", "page et al", "prisma 2020", "ieee tsg", "examples cont", "redaction policy")):
                continue
                
            counts[key] += 1
            display.setdefault(key, clean_phrase)

    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [Concept(name=display[key], frequency=counts[key]) for key in ranked[:max_concepts]]


_CONNECTING_WORDS = frozenset(
    {"with", "in", "of", "for", "and", "to", "on", "at", "by", "the", "a", "an", "from", "into", "as"}
)


def extract_main_concept(pages: list[Page] | None = None, concepts: list[Concept] | None = None) -> str | None:
    """Identify the overarching lecture topic / main concept from the title slide or concepts."""
    if pages and len(pages) > 0:
        p1 = pages[0].text
        lines = [line.strip() for line in p1.splitlines() if line.strip()]
        candidate_parts: list[str] = []
        for line in lines[:8]:
            lower = line.casefold()
            # Stop if course code, instructor, or administration line is reached
            if re.search(r"\b[a-z]{2,4}\d{4}\b", lower) or "–" in line or " - " in line or any(m in lower for m in ("administration", "storage", "course", "unit", "it4011", "it4041", "indika", "alahapperuma", "liyanage", "department", "faculty", "university")):
                break
            if re.match(r"^\d{1,2}$", line):
                continue
            # If the line ends with a period, it is body text, not a slide title
            if line.endswith(".") and not line.endswith("etc."):
                continue
            if lower.startswith("course:"):
                raw = line.split(":", 1)[1].strip()
                return _format_title(raw)
            if lower.startswith("unit 1:") or lower.startswith("unit:"):
                raw = line.split(":", 1)[1].strip()
                return _format_title(raw)
            candidate_parts.append(line)
            if len(candidate_parts) >= 2:
                last_word = candidate_parts[-1].split()[-1].casefold() if candidate_parts[-1].split() else ""
                if last_word not in _CONNECTING_WORDS:
                    break
                if len(candidate_parts) >= 4:
                    break
        
        if candidate_parts:
            combined = " ".join(candidate_parts)
            title = re.sub(r"\s+", " ", combined).strip()
            if 5 <= len(title) <= 65:
                return _format_title(title)

    if concepts and len(concepts) > 0:
        return _format_title(concepts[0].name)
    return None


def _format_title(title: str) -> str:
    words = title.split()
    return " ".join(w.capitalize() if w.lower() not in STOP_WORDS else w.lower() for w in words)
