import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_fit_proper():
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
        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            async def call(m, p={}):
                await ws.send(json.dumps({'id': 1, 'method': m, 'params': p}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get('id') == 1: return d.get('result', {})
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)
            
            # Let's test standard physics = false, hierarchical with fixed level separation, and fit()
            js = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                const container = document.getElementById('graph-canvas');
                
                const visNodes = data.graph.nodes.map(n => ({{
                  id: n.id,
                  label: n.label,
                  shape: 'box',
                  margin: 10,
                  font: {{ color: '#ffffff', size: 13, bold: true }},
                  color: {{
                    background: '#0284c7',
                    border: '#38bdf8',
                    highlight: {{ background: '#38bdf8', border: '#ffffff' }}
                  }}
                }}));

                const visEdges = data.graph.edges.map(e => ({{
                  from: e.source,
                  to: e.target,
                  arrows: 'to',
                  color: {{ color: '#38bdf8', opacity: 0.8 }},
                  width: 2
                }}));

                // Clean options WITHOUT physics fighting the camera
                const options = {{
                  layout: {{
                    hierarchical: {{
                      enabled: true,
                      direction: 'UD',
                      sortMethod: 'hubsize',
                      levelSeparation: 120,
                      nodeSpacing: 180
                    }}
                  }},
                  physics: false,
                  interaction: {{ hover: true, zoomView: true, dragView: true }}
                }};

                if (network) network.destroy();
                network = new vis.Network(container, {{ nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) }}, options);
                
                // Immediately compute bounding box of all nodes and center camera
                network.fit();

                document.getElementById('canvas-placeholder').style.display = 'none';
                document.getElementById('canvas-toolbar').style.display = 'flex';
                
                return {{
                    view: network.getViewPosition(),
                    scale: network.getScale(),
                    firstNode: network.getPositions()[data.graph.nodes[0].id]
                }};
            }})()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("AFTER SYNC FIT:", r.get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('real_render_screenshot_fitted.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('data')))
            print("Saved real_render_screenshot_fitted.png")
    finally:
        proc.terminate()

asyncio.run(test_fit_proper())
