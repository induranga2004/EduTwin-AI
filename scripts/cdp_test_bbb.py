import asyncio
import json
import os
import subprocess
import time
import urllib.request
import websockets
import base64

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222

async def run_cdp_test():
    # 1. Get join URL
    secret = "X87g_4v8m5a8v_9"  # let's get it or use dev.ps1
    res = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-Command", r".\scripts\dev.ps1 meeting -NoOpen"],
        capture_output=True,
        text=True,
    )
    join_url = None
    for line in res.stdout.splitlines():
        if "https://bbb.test/bigbluebutton/api/join?" in line:
            join_url = line.strip()
            break
    
    print(f"[+] Fresh Join URL: {join_url}")
    if not join_url:
        print("Failed to get join URL")
        return

    # 2. Launch Chrome with CDP
    proc = subprocess.Popen([
        CHROME_PATH,
        f"--remote-debugging-port={PORT}",
        "--headless=new",
        "--ignore-certificate-errors",
        "--window-size=1920,1080",
        "--disable-gpu",
        "--no-sandbox",
        "about:blank"
    ])
    
    try:
        # Wait for port to open
        ws_url = None
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")
                targets = json.loads(resp.read().decode())
                if targets:
                    ws_url = targets[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
        
        if not ws_url:
            print("Failed to connect to Chrome CDP")
            return
            
        print(f"[+] Connected to CDP: {ws_url}")
        
        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            msg_id = 1
            async def send_cmd(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                payload = {"id": cid, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("id") == cid:
                        return data.get("result", {})

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            
            print("[+] Navigating to BBB join URL...")
            await send_cmd("Page.navigate", {"url": join_url})
            
            # Wait for meeting room to load (wait 15 seconds)
            print("[+] Waiting 15s for BigBlueButton to load past the connecting screen...")
            for s in range(15):
                await asyncio.sleep(1)
                # Check status
                eval_res = await send_cmd("Runtime.evaluate", {
                    "expression": "({ url: location.href, title: document.title, hasBtn: !!document.getElementById('edutwin-trigger-btn'), btnVisible: document.getElementById('edutwin-trigger-btn') ? (document.getElementById('edutwin-trigger-btn').offsetWidth > 0) : false })",
                    "returnByValue": True
                })
                val = eval_res.get("result", {}).get("value", {})
                print(f"    [{s+1}s] URL: {val.get('url')} | HasBtn: {val.get('hasBtn')} | BtnVisible: {val.get('btnVisible')}")

            # Get detailed bounding rect and computed styles
            detail_res = await send_cmd("Runtime.evaluate", {
                "expression": """
                (() => {
                    const btn = document.getElementById('edutwin-trigger-btn');
                    if (!btn) return { error: 'No button found' };
                    const rect = btn.getBoundingClientRect();
                    const style = window.getComputedStyle(btn);
                    return {
                        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height, top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left },
                        zIndex: style.zIndex,
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        parent: btn.parentElement ? btn.parentElement.tagName : null,
                        bodyChildrenCount: document.body.children.length
                    };
                })()
                """,
                "returnByValue": True
            })
            print("[+] Button Details:", json.dumps(detail_res.get("result", {}).get("value"), indent=2))

            # Take screenshot
            shot_res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            img_b64 = shot_res.get("data")
            if img_b64:
                out_path = r"C:\Users\Hiru\Desktop\Research\c1\EduTwin-AI\bbb_meeting_live.png"
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                print(f"[+] Saved live meeting screenshot to {out_path} ({len(img_b64)} b64 bytes)")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(run_cdp_test())
