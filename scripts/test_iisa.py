import asyncio, json, urllib.request, websockets, subprocess, base64

async def test_iisa():
    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9999', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9999/json')
        ws_url = [t for t in json.loads(resp.read().decode()) if t.get('type') == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url) as ws:
            async def call(m, p={}):
                await ws.send(json.dumps({'id': 1, 'method': m, 'params': p}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get('id') == 1: return d.get('result', {})
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)
            
            # Run the EXACT sample IISA graph that viva_dashboard had originally
            js = """
            (() => {
                const SAMPLE_IISA_GRAPH = {
                  nodes: [
                    { id: "intro_infosec", label: "Introduction to\\nInformation Security", description: "Foundational principles.", depth: 0, status: "not_yet_observed" },
                    { id: "crypto_basics", label: "Cryptography\\nBasics", description: "Foundational cryptosystem.", depth: 1, status: "not_yet_observed" },
                    { id: "symmetric_encryption", label: "Symmetric\\nEncryption", description: "Cryptographic primitives.", depth: 2, status: "not_yet_observed" }
                  ],
                  edges: [
                    { from: "intro_infosec", to: "crypto_basics", label: "prerequisite" },
                    { from: "crypto_basics", to: "symmetric_encryption", label: "prerequisite" }
                  ]
                };
                initGraph(SAMPLE_IISA_GRAPH);
                document.getElementById('canvas-placeholder').style.display = 'none';
                return 'LOADED SAMPLE';
            })()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("SAMPLE LOADED:", r.get('result', {}).get('value'))
            await asyncio.sleep(2)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('sample_iisa_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('data')))
            print("Saved sample_iisa_screenshot.png")
    finally:
        proc.terminate()

asyncio.run(test_iisa())
