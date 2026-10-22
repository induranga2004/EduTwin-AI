import asyncio
import json
import urllib.request
import websockets
import subprocess
import httpx
import base64

async def test_live_dashboard():
    pptx_path = r'hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx'
    print(f"[1] Testing upload with {pptx_path}...")
    with open(pptx_path, 'rb') as f:
        files = {'file': ('lecture.pptx', f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
        with httpx.Client(verify=False, timeout=120.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()
    print(f"Backend response: {backend_data.get('concept_count')} concepts, {len(backend_data.get('graph', {}).get('nodes', []))} nodes, {len(backend_data.get('graph', {}).get('edges', []))} edges")

    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9777', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9777/json')
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
            
            print("[2] Loading https://bbb.test/edusync/api/c1/dashboard in Chrome...")
            await send_cmd('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)

            # Check initial state (should have NO IISA nodes, placeholder visible)
            initial_state = await send_cmd('Runtime.evaluate', {
                'expression': """
                ({
                    placeholderDisplay: window.getComputedStyle(document.getElementById('canvas-placeholder')).display,
                    networkExists: !!network,
                    nodeCount: nodesDataset ? nodesDataset.length : 0,
                    btnDisabled: document.getElementById('build-graph-btn').disabled
                })
                """,
                'returnByValue': True
            })
            print("[3] Initial State (Zero IISA nodes loaded by default):", repr(initial_state.get('result', {}).get('value')))

            # Now simulate building the graph with the uploaded PPTX data
            print("[4] Simulating graph build from uploaded PPTX...")
            build_js = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                stagedFile = {{ name: 'lecture.pptx', size: 24239570 }};
                
                document.getElementById('active-slide-badge').innerText = stagedFile.name;
                document.getElementById('course-label').innerText = data.concept_count + ' Concepts Extracted';

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

                return {{
                    stats: document.getElementById('graph-stats').innerText,
                    nodeCount: nodesDataset.length,
                    edgeCount: edgesDataset.length,
                    networkBodyNodes: network.body.nodeIndices.length,
                    viewPosition: network.getViewPosition(),
                    scale: network.getScale()
                }};
            }})()
            """
            build_res = await send_cmd('Runtime.evaluate', {'expression': build_js, 'returnByValue': True})
            print("[5] Graph Rendered State:", repr(build_res.get('result', {}).get('value')))

            # Take screenshot of the rendered PPTX graph
            shot_res = await send_cmd('Page.captureScreenshot', {'format': 'png'})
            img_b64 = shot_res.get('data')
            if img_b64:
                out_path = r'C:\Users\Hiru\Desktop\Research\c1\EduTwin-AI\pptx_graph_screenshot.png'
                with open(out_path, 'wb') as f:
                    f.write(base64.b64decode(img_b64))
                print(f"[6] Saved screenshot to {out_path} ({len(img_b64)} b64 bytes)")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    asyncio.run(test_live_dashboard())
