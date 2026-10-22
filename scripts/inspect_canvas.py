import asyncio, json, urllib.request, websockets, subprocess, base64

async def check():
    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9999', '--headless=new', '--ignore-certificate-errors'])
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
            
            # Call initGraph
            js = """
            (() => {
                const testGraph = {
                  nodes: [
                    { id: '1', label: 'Concept 1', depth: 0, status: 'not_yet_observed' },
                    { id: '2', label: 'Concept 2', depth: 1, status: 'not_yet_observed' }
                  ],
                  edges: [{ from: '1', to: '2' }]
                };
                initGraph(testGraph);
                document.getElementById('canvas-placeholder').style.display = 'none';
                return {
                    innerHTML: document.getElementById('graph-canvas').innerHTML.substring(0, 300),
                    canvasStyle: window.getComputedStyle(document.getElementById('graph-canvas')).display,
                    networkBodyNodes: network.body.nodeIndices
                };
            })()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("AFTER INIT:", r.get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            b64 = shot.get('data')
            if b64:
                with open('test_2nodes_screenshot.png', 'wb') as f:
                    f.write(base64.b64decode(b64))
                print("Saved test_2nodes_screenshot.png")
    finally:
        proc.terminate()

asyncio.run(check())
