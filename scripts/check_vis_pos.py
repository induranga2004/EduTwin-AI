import asyncio, json, urllib.request, websockets, subprocess, httpx

async def check_vis_coordinates():
    with open(r'hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx', 'rb') as f:
        files = {'file': ('lecture.pptx', f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
        with httpx.Client(verify=False, timeout=120.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9666', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9666/json')
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
            
            # Run initGraph
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
                // Return node positions
                const pos = network.getPositions();
                const viewPos = network.getViewPosition();
                const scale = network.getScale();
                return {{ pos: pos, viewPos: viewPos, scale: scale }};
            }})()
            """
            eval_res = await send_cmd('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            val = eval_res.get('result', {}).get('value', {})
            print("View Position:", val.get('viewPos'))
            print("Scale:", val.get('scale'))
            pos = val.get('pos', {})
            print("Number of positioned nodes:", len(pos))
            for k in list(pos.keys())[:5]:
                print(f"  Node {k}: {pos[k]}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    asyncio.run(check_vis_coordinates())
