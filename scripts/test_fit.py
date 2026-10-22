import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_fit():
    with open(r'hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx', 'rb') as f:
        files = {'file': ('lecture.pptx', f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
        with httpx.Client(verify=False, timeout=120.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9999', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9999/json')
        targets = [t for t in json.loads(resp.read().decode()) if t.get('type') == 'page']
        ws_url = targets[0]['webSocketDebuggerUrl']
        
        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            msg_id = 1
            async def send_cmd(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                await ws.send(json.dumps({'id': cid, 'method': method, 'params': params or {}}))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get('id') == cid: return data.get('result', {})

            await send_cmd('Page.enable')
            await send_cmd('Runtime.enable')
            await send_cmd('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)
            
            # Execute initGraph with BarnesHut or compact layout and network.fit()
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
                
                const container = document.getElementById('graph-canvas');
                const visNodes = graphData.nodes.map(n => ({{
                  id: n.id,
                  label: n.label,
                  shape: 'box',
                  margin: 10,
                  font: {{ color: '#ffffff', size: 13, bold: true }},
                  color: {{
                    background: '#334155',
                    border: '#38bdf8',
                    highlight: {{ background: '#0284c7', border: '#ffffff' }}
                  }}
                }}));

                const visEdges = graphData.edges.map(e => ({{
                  from: e.from,
                  to: e.to,
                  arrows: 'to',
                  color: {{ color: '#38bdf8', opacity: 0.8 }},
                  width: 2
                }}));

                // Force-directed layout or relaxed physics
                const options = {{
                  physics: {{
                    enabled: true,
                    barnesHut: {{
                      gravitationalConstant: -3000,
                      centralGravity: 0.3,
                      springLength: 120,
                      springConstant: 0.04,
                      damping: 0.09
                    }},
                    stabilization: {{
                      enabled: true,
                      iterations: 150
                    }}
                  }},
                  interaction: {{ hover: true, zoomView: true, dragView: true }}
                }};

                if (network) network.destroy();
                network = new vis.Network(container, {{ nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) }}, options);
                
                // On stabilization, fit camera!
                network.once('stabilizationIterationsDone', function() {{
                  network.setOptions({{ physics: false }});
                  network.fit();
                }});

                // Also force fit after 500ms
                setTimeout(() => {{
                  network.fit();
                }}, 500);

                document.getElementById('canvas-placeholder').style.display = 'none';
                document.getElementById('canvas-toolbar').style.display = 'flex';
                
                return 'INITIALIZED';
            }})()
            """
            await send_cmd('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            await asyncio.sleep(2)
            
            fit_check = await send_cmd('Runtime.evaluate', {
                'expression': '({ view: network.getViewPosition(), scale: network.getScale(), box: network.getBoundingBox(network.body.nodeIndices[0]) })',
                'returnByValue': True
            })
            print("[AFTER FIT]:", fit_check.get('result', {}).get('value'))

            shot = await send_cmd('Page.captureScreenshot', {'format': 'png'})
            b64 = shot.get('data')
            if b64:
                with open('pptx_graph_screenshot_fitted.png', 'wb') as f:
                    f.write(base64.b64decode(b64))
                print("Saved pptx_graph_screenshot_fitted.png")

    finally:
        proc.terminate()
        proc.wait()

asyncio.run(test_fit())
