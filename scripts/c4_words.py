"""C4 slide-word helper. Runs INSIDE the edusync container (see scripts/c4-words.ps1).

Reads the output of `pdftotext -bbox` (word positions of a PDF, in PDF points) and stores the words
of each page in C4 for the matching slide, converted to whiteboard coordinates. C4 uses them to
name the words an underline or circle points at. Standard library only; ASCII output.

  python /tmp/c4_words.py --lecture <meetingId> --xml /tmp/c4words.html
"""

import argparse
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

C4 = "http://127.0.0.1:8100/api/c4"
PAGE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)
WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', re.S
)


def http_json(method, url, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]


def fail(message):
    print("ERROR: " + message)
    sys.exit(1)


def parse_pages(text):
    """[(page_width, page_height, [(word, xmin, ymin, xmax, ymax), ...]), ...] in page order."""
    pages = []
    for match in PAGE.finditer(text):
        words = [
            (html.unescape(w[4]).strip(), float(w[0]), float(w[1]), float(w[2]), float(w[3]))
            for w in WORD.findall(match.group(3))
        ]
        pages.append((float(match.group(1)), float(match.group(2)), [w for w in words if w[0]]))
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lecture", required=True)
    parser.add_argument("--xml", required=True)
    args = parser.parse_args()

    with open(args.xml, encoding="utf-8") as handle:
        pages = parse_pages(handle.read())
    if not pages:
        fail("No pages found in the pdftotext output.")

    status, slides = http_json("GET", "%s/lectures/%s/slides" % (C4, args.lecture))
    if status != 200 or not slides:
        fail("C4 has no slides for this lecture yet. Run scripts\\c4-capture.ps1 first.")
    by_number = {s["number"]: s for s in slides if s.get("number") is not None}

    stored = skipped = total = 0
    for number, (page_w, page_h, words) in enumerate(pages, start=1):
        slide = by_number.get(number)
        if not slide or not slide.get("scaled_width") or not slide.get("scaled_height") or not words:
            skipped += 1
            continue
        sx = slide["scaled_width"] / page_w
        sy = slide["scaled_height"] / page_h
        body = {
            "words": [
                {
                    "text": text[:200],
                    "x": round(x0 * sx, 2),
                    "y": round(y0 * sy, 2),
                    "width": round((x1 - x0) * sx, 2),
                    "height": round((y1 - y0) * sy, 2),
                }
                for text, x0, y0, x1, y1 in words[:5000]
            ]
        }
        slide_path = urllib.parse.quote(slide["slide_id"], safe="/")
        code, resp = http_json(
            "PUT", "%s/lectures/%s/slides/%s/words" % (C4, args.lecture, slide_path), body
        )
        if code != 200:
            print("slide %d FAILED (%s): %s" % (number, code, str(resp)[:150]))
            continue
        stored += 1
        total += len(body["words"])
    print(
        "word positions stored for %d of %d slides (%d words); %d pages skipped (no text or no slide)"
        % (stored, len(pages), total, skipped)
    )


if __name__ == "__main__":
    main()
