import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_eval():
    with open(r'hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx', 'rb') as f:
        files = {'file': ('lecture.pptx', f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
        with httpx.Client(verify=False, timeout=120.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9888', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9888/json')
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
                    if data.get('id') == cid:
                        return data
                    if 'error' in data and data.get('id') == cid:
                        return data

            await send_cmd('Page.enable')
            await send_cmd('Runtime.enable')
            await send_cmd('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)
            
            # Test simple eval
            r1 = await send_cmd('Runtime.evaluate', {'expression': '1 + 1', 'returnByValue': True})
            print("Simple eval:", r1)
            
            # Now simulate building graph
            js = f"""
            (() => {{
                try {{
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
                    document.getElementById('canvas-placeholder').style.display = 'none';
                    document.getElementById('canvas-toolbar').style.display = 'flex';
                    centerGraphView();
                    return 'SUCCESS: ' + graphData.nodes.length + ' nodes';
                }} catch (e) {{
                    return 'ERROR: ' + e.toString() + ' | ' + e.stack;
                }}
            }})()
            """
            r2 = await send_cmd('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("Graph build eval result:", r2)
            await asyncio.sleep(1)

            shot = await send_cmd('Page.captureScreenshot', {'format': 'png'})
            b64 = shot.get('result', {}).get('data')
            if b64:
                with open('pptx_graph_screenshot_rendered.png', 'wb') as f:
                    f.write(base64.b64decode(b64))
                print("Saved pptx_graph_screenshot_rendered.png")

    finally:
        proc.terminate()
        proc.wait()

asyncio.run(test_eval())
