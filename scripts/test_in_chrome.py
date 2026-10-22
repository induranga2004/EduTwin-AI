import asyncio
import json
import subprocess
import urllib.request
import websockets
import base64

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333

async def run_test():
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
        await asyncio.sleep(1)
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
                    if data.get("method") == "Runtime.consoleAPICalled":
                        print("    [BROWSER CONSOLE]", data.get("params", {}).get("type"), [a.get("value") for a in data.get("params", {}).get("args", [])])

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            
            print("[+] Navigating to Dashboard...")
            await send_cmd("Page.navigate", {"url": "https://bbb.test/edusync/api/c1/dashboard"})
            await asyncio.sleep(2)
            
            # Now let's fetch /edusync/api/c1/lectures/graph inside the browser or test initGraph with the response
            print("[+] Calling fetch in browser context...")
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": """
                (async () => {
                    try {
                        console.log('Fetching sample or testing initGraph...');
                        // Let's test calling initGraph with a graph that has cyclic/undirected edges
                        const testGraph = {
                            nodes: [
                                { id: "a", label: "Concept A", depth: 1, status: "not_yet_observed" },
                                { id: "b", label: "Concept B", depth: 1, status: "not_yet_observed" },
                                { id: "c", label: "Concept C", depth: 1, status: "not_yet_observed" }
                            ],
                            edges: [
                                { from: "a", to: "b" },
                                { from: "b", to: "c" },
                                { from: "c", to: "a" } // CYCLE!
                            ]
                        };
                        initGraph(testGraph);
                        return { success: true };
                    } catch (e) {
                        console.error('Error during initGraph:', e.toString());
                        return { error: e.toString(), stack: e.stack };
                    }
                })()
                """,
                "awaitPromise": True,
                "returnByValue": True
            })
            print("[+] Result with cyclic edges:", eval_res)
            
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(run_test())
