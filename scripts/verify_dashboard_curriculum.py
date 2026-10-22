import asyncio
import json
import subprocess
import urllib.request
import ssl
import websockets
import base64
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9334
PDF_PATH = r"hiru backend\data\uploads\8b0315c3-b1de-48b4-b2cd-70a02ac350d8.pdf"

async def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("[+] Requesting graph from backend via BBB proxy...")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="Database_Security.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "https://127.0.0.1/edusync/api/c1/lectures/graph",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req, context=ctx) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))

    graph_data = res_data["graph"]
    print(f"[+] Received graph: {graph_data['node_count']} nodes, {graph_data['edge_count']} edges")
    print(f"[+] Main concept: {graph_data.get('main_concept')}")
    print(f"[+] Root node: {graph_data['nodes'][0]['label']}")

    # 2. Launch Chrome
    print("[+] Launching Chrome...")
    proc = subprocess.Popen([
        CHROME_PATH,
        f"--remote-debugging-port={PORT}",
        "--headless=new",
        "--ignore-certificate-errors",
        "--window-size=1920,1080",
        "--disable-gpu",
        "about:blank"
    ])

    try:
        await asyncio.sleep(1.5)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")
        targets = json.loads(resp.read().decode())
        ws_url = targets[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            msg_id = 1
            async def send_cmd(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                await ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("id") == cid:
                        return data.get("result", {})

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            print("[+] Navigating to Dashboard...")
            await send_cmd("Page.navigate", {"url": "https://127.0.0.1/edusync/api/c1/dashboard"})
            await asyncio.sleep(2)

            print("[+] Injecting graphData into dashboard...")
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": f"""
                (() => {{
                    const gd = {json.dumps(graph_data)};
                    window.currentGraphData = gd;
                    window.initGraph(gd);
                    return {{
                        rendered: !!document.getElementById("svg-root"),
                        stats: document.getElementById("graph-stats").innerText,
                        rootNodeTag: document.querySelector(".concept-node text")?.textContent,
                        nodesCount: document.querySelectorAll(".concept-node").length
                    }};
                }})()
                """,
                "returnByValue": True
            })

            res_val = eval_res.get("result", {}).get("value", {})
            print("[+] In-browser evaluation result:", res_val)

            await asyncio.sleep(1)
            print("[+] Capturing screenshot...")
            shot = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_data = shot.get("data", "")
            if b64_data:
                Path("verified_curriculum_graph.png").write_bytes(base64.b64decode(b64_data))
                print("[+] Saved verified_curriculum_graph.png successfully!")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
