import re

with open("bbb_dom.html", "r", encoding="utf-8") as f:
    text = f.read()

title = re.search(r"<title>(.*?)</title>", text)
print("Title:", title.group(1) if title else "None")

app_match = re.search(r'<div id="app"[^>]*>(.*?)</div>', text, re.DOTALL)
if app_match:
    print("App content length:", len(app_match.group(1)))
    print("App snippet:", repr(app_match.group(1)[:500]))

# Check what scripts are loaded or any error messages
scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', text)
print("Script src tags:", scripts)

body_text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
body_text = re.sub(r'<script.*?</script>', '', body_text, flags=re.DOTALL)
body_text = re.sub(r'<[^>]+>', ' ', body_text)
body_text = ' '.join(body_text.split())
print("Visible text on page:", body_text[:500])
