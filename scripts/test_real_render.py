import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_real_render():
    with open(r'hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx', 'rb') as f:
        files = {'file': ('lecture.pptx', f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
        with httpx.Client(verify=False, timeout=120.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

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
            
            # Hide placeholder, hide loading, call initGraph
            js = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                const graphData = {{
                  nodes: data.graph.nodes.map(n => ({{
                    id: n.id,
                    label: n.label,
                    description: n.description || '',
                    depth: n.depth !== undefined ? n.depth : 1,
                    status: 'not_yet_observed'
                  }})),
                  edges: data.graph.edges.map(e => ({{
                    from: e.source,
                    to: e.target
                  }}))
                }};
                
                document.getElementById('canvas-placeholder').style.display = 'none';
                document.getElementById('canvas-loading').style.display = 'none';
                document.getElementById('canvas-toolbar').style.display = 'flex';
                
                initGraph(graphData);
                
                return {{
                    nodeCount: graphData.nodes.length,
                    edgeCount: graphData.edges.length,
                    positions: Object.keys(network.getPositions()).length,
                    view: network.getViewPosition(),
                    scale: network.getScale()
                }};
            }})()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("INIT RESULT:", r.get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('real_render_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('data')))
            print("Saved real_render_screenshot.png")
    finally:
        proc.terminate()

asyncio.run(test_real_render())
