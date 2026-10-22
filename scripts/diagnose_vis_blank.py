import asyncio, json, urllib.request, websockets, subprocess, httpx

async def diagnose():
    # Use the real uploaded pptx
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
            
            # Execute initGraph and diagnose everything
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
                
                initGraph(graphData);
                
                const container = document.getElementById('graph-canvas');
                const canvas = container.querySelector('canvas');
                const ctx = canvas ? canvas.getContext('2d') : null;
                
                const pos = network.getPositions();
                const view = network.getViewPosition();
                const scale = network.getScale();
                const bBox = network.getBoundingBox(network.body.nodeIndices[0]);

                return {{
                    containerWidth: container.clientWidth,
                    containerHeight: container.clientHeight,
                    canvasWidth: canvas ? canvas.width : 0,
                    canvasHeight: canvas ? canvas.height : 0,
                    nodeCount: graphData.nodes.length,
                    edgeCount: graphData.edges.length,
                    positionsCount: Object.keys(pos).length,
                    firstNodePos: pos[Object.keys(pos)[0]],
                    viewPosition: view,
                    scale: scale,
                    sampleBoundingBox: bBox
                }};
            }})()
            """
            eval_res = await send_cmd('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("[DIAGNOSIS RESULT]:", json.dumps(eval_res.get('result', {}).get('value'), indent=2))

    finally:
        proc.terminate()
        proc.wait()

asyncio.run(diagnose())
