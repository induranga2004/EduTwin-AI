"""C4 capture helper. Runs INSIDE the edusync container (see scripts/c4-capture.ps1).

Reads whiteboard annotations and slide info from BigBlueButton's Hasura and stores them in C4
through its HTTP API (127.0.0.1:8100). Standard library only; ASCII only; never prints secrets.

  --info             print the running meeting and its presentation as JSON
  --slides DIR       store slide info, text and the slide images found in DIR (slideN.svg)
  --once             capture the annotations of the slide currently shown (idempotent)

BBB only exposes annotation history for the CURRENT slide, so --once must be called repeatedly
during the lecture. Re-importing the same events is harmless (duplicates are ignored).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

C4 = "http://127.0.0.1:8100/api/c4"


def http_json(method, url, body=None, headers=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]


def hasura(gql):
    url = os.environ.get("EDUSYNC_HASURA_URL", "")
    secret = os.environ.get("EDUSYNC_HASURA_ADMIN_SECRET", "")
    if not url or not secret:
        sys.exit("EDUSYNC_HASURA_URL / EDUSYNC_HASURA_ADMIN_SECRET are not set in the container")
    status, body = http_json("POST", url, {"query": gql}, {"x-hasura-admin-secret": secret})
    if status != 200 or (isinstance(body, dict) and body.get("errors")):
        sys.exit("Hasura query failed (%s): %s" % (status, str(body)[:300]))
    return body["data"]


def current_meeting():
    meetings = hasura("query { meeting { meetingId extId name } }")["meeting"]
    if not meetings:
        sys.exit("No running meeting found. Start one with .\\scripts\\dev.ps1 meeting")
    # newest meeting: its id ends with the creation time in epoch milliseconds
    return max(meetings, key=lambda m: int(m["meetingId"].rsplit("-", 1)[-1] or 0))


def start_ms(meeting_id):
    """Lecture time zero = meeting creation time (the id suffix), stable across calls."""
    return float(meeting_id.rsplit("-", 1)[-1])


def pages(meeting_id):
    gql = (
        "query { pres_page { pageId num width height scaledWidth scaledHeight content "
        "isCurrentPage presentationId } }"
    )
    return hasura(gql)["pres_page"]


def current_presentation(rows):
    for row in rows:
        if row.get("isCurrentPage"):
            return row["presentationId"]
    return rows[0]["presentationId"] if rows else None


def cmd_info():
    meeting = current_meeting()
    rows = pages(meeting["meetingId"])
    print(
        json.dumps(
            {
                "meetingId": meeting["meetingId"],
                "extId": meeting["extId"],
                "presentationId": current_presentation(rows),
                "pages": len(rows),
            }
        )
    )


def cmd_slides(directory):
    meeting = current_meeting()
    meeting_id = meeting["meetingId"]
    rows = pages(meeting_id)
    presentation_id = current_presentation(rows)
    stored = with_svg = 0
    for row in rows:
        if row["presentationId"] != presentation_id:
            continue
        body = {
            "number": row["num"],
            "width": row["width"],
            "height": row["height"],
            "scaled_width": row["scaledWidth"],
            "scaled_height": row["scaledHeight"],
            "text": row["content"],
        }
        path = os.path.join(directory, "slide%s.svg" % row["num"])
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                body["svg"] = handle.read()
        body = {k: v for k, v in body.items() if v is not None}
        slide = urllib.parse.quote(row["pageId"], safe="/")
        status, resp = http_json("PUT", "%s/lectures/%s/slides/%s" % (C4, meeting_id, slide), body)
        if status != 200:
            print("slide %s FAILED (%s): %s" % (row["num"], status, str(resp)[:200]))
            continue
        stored += 1
        with_svg += 1 if resp.get("has_svg") else 0
    print("slides stored: %d of %d (with image: %d)" % (stored, len(rows), with_svg))


def cmd_once():
    meeting = current_meeting()
    meeting_id = meeting["meetingId"]
    gql = (
        "query { pres_annotation_history_curr(order_by: {updatedAt: asc}) "
        "{ meetingId pageId userId annotationId annotationInfo updatedAt } }"
    )
    rows = [
        r for r in hasura(gql)["pres_annotation_history_curr"] if r["meetingId"] == meeting_id
    ]
    if not rows:
        print("no annotations on the current slide yet")
        return
    body = {
        "lecture_id": meeting_id,
        "slide_id": rows[0]["pageId"],
        "events": rows,
        "lecture_start_ms": start_ms(meeting_id),
    }
    status, resp = http_json("POST", C4 + "/events/import", body)
    if status != 200:
        print("import FAILED (%s): %s" % (status, str(resp)[:300]))
        sys.exit(1)
    print(
        "slide %s: %d rows, %d new, %d already stored, %d unplaced"
        % (
            rows[0]["pageId"].rsplit("/", 1)[-1],
            len(rows),
            resp["imported_count"],
            resp["duplicate_count"],
            resp["unplaced_count"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--info", action="store_true")
    group.add_argument("--slides", metavar="DIR")
    group.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.info:
        cmd_info()
    elif args.slides:
        cmd_slides(args.slides)
    else:
        cmd_once()


if __name__ == "__main__":
    main()
