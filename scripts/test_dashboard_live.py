import asyncio
import json
import subprocess
import urllib.request
import websockets
import base64
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9337

def get_graph_data():
    print("[+] Extracting graph data from backend...")
    res = subprocess.run([
        "docker", "exec", "edutwin-ai-edusync-1", "python", "-c",
        """
import json, httpx
with open('/tmp/user1.pdf', 'rb') as f:
    res = httpx.post('http://localhost:8100/api/c1/lectures/graph', files={'file': ('user1.pdf', f, 'application/pdf')}, timeout=30.0)
print(json.dumps(res.json()['graph']))
"""
    ], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip())

async def main():
    graph_data = get_graph_data()
    print(f"[+] Loaded graph: {graph_data['node_count']} nodes, {graph_data['edge_count']} edges")
    print(f"[+] Root node: {graph_data['nodes'][0]['label']}")

    print("[+] Launching Chrome on port", PORT)
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

        async with websockets.connect(ws_url, max_size=None) as ws:
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

            print("[+] Navigating to Dashboard at https://bbb.test/edusync/api/c1/dashboard ...")
            await send_cmd("Page.navigate", {"url": "https://bbb.test/edusync/api/c1/dashboard"})
            await asyncio.sleep(2)

            print("[+] Rendering graph into SVG canvas...")
            gd_json = json.dumps(graph_data)
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": f"""
                (() => {{
                    const gd = {gd_json};
                    window.currentGraph = gd;
                    window.initGraph(gd);
                    return {{
                        rendered: !!document.getElementById("svg-root"),
                        stats: document.getElementById("graph-stats").innerText,
                        rootNodeLabel: document.querySelector(".concept-node text:last-child")?.textContent,
                        renderedNodes: document.querySelectorAll(".concept-node").length
                    }};
                }})()
                """,
                "returnByValue": True
            })

            res_val = eval_res.get("result", {}).get("value", {})
            print("[+] SVG Render result:", json.dumps(res_val, indent=2))

            await asyncio.sleep(1)
            print("[+] Capturing screenshot...")
            shot = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_data = shot.get("data", "")
            if b64_data:
                Path("verified_clean_graph.png").write_bytes(base64.b64decode(b64_data))
                print("[+] Saved verified_clean_graph.png successfully!")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
