import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_svg():
    # Fetch real 30-node PDF data from backend
    with open(r'hiru backend\data\uploads\8b0315c3-b1de-48b4-b2cd-70a02ac350d8.pdf', 'rb') as f:
        files = {'file': ('lecture.pdf', f, 'application/pdf')}
        with httpx.Client(verify=False, timeout=60.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()
    print("Backend data loaded:", backend_data.get('concept_count'), "concepts")

    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9998', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9998/json')
        ws_url = [t for t in json.loads(resp.read().decode()) if t.get('type') == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url) as ws:
            async def call(m, p={}):
                await ws.send(json.dumps({'id': 1, 'method': m, 'params': p}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get('id') == 1: return d
            await call('Runtime.enable')
            await call('Page.enable')
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(1)
            
            # Inject SVG graph renderer into #graph-canvas
            js = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                const container = document.getElementById('graph-canvas');
                document.getElementById('canvas-placeholder').style.display = 'none';
                document.getElementById('canvas-toolbar').style.display = 'flex';
                document.getElementById('course-label').innerText = data.concept_count + ' Concepts Extracted';
                document.getElementById('graph-stats').innerText = data.concept_count + ' Nodes | ' + data.graph.edges.length + ' Prerequisite Edges';

                // Organize nodes by depth
                const depthGroups = {{}};
                data.graph.nodes.forEach(n => {{
                    const d = n.depth !== undefined ? n.depth : 0;
                    if (!depthGroups[d]) depthGroups[d] = [];
                    depthGroups[d].push(n);
                }});

                const nodeCoords = {{}};
                const nodeWidth = 200;
                const nodeHeight = 56;
                const levelYSpacing = 140;
                const nodeXSpacing = 240;

                let minX = 99999, maxX = -99999, minY = 99999, maxY = -99999;

                Object.keys(depthGroups).forEach(d => {{
                    const group = depthGroups[d];
                    const levelY = 60 + parseInt(d) * levelYSpacing;
                    const totalWidth = group.length * nodeXSpacing;
                    const startX = 700 - totalWidth / 2;

                    group.forEach((node, idx) => {{
                        const x = startX + idx * nodeXSpacing;
                        const y = levelY;
                        nodeCoords[node.id] = {{ x, y, node }};
                        minX = Math.min(minX, x);
                        maxX = Math.max(maxX, x + nodeWidth);
                        minY = Math.min(minY, y);
                        maxY = Math.max(maxY, y + nodeHeight);
                    }});
                }});

                // Build SVG
                let svgHtml = `
                <svg id="graph-svg" width="100%" height="100%" style="background: #090d16; cursor: grab; user-select: none;">
                  <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
                    </marker>
                  </defs>
                  <g id="svg-viewport">
                `;

                // Draw Edges
                data.graph.edges.forEach(edge => {{
                    const s = nodeCoords[edge.source];
                    const t = nodeCoords[edge.target];
                    if (s && t) {{
                        const x1 = s.x + nodeWidth / 2;
                        const y1 = s.y + nodeHeight;
                        const x2 = t.x + nodeWidth / 2;
                        const y2 = t.y;
                        const cy1 = y1 + 40;
                        const cy2 = y2 - 40;
                        svgHtml += `<path d="M ${{x1}} ${{y1}} C ${{x1}} ${{cy1}}, ${{x2}} ${{cy2}}, ${{x2}} ${{y2}}" fill="none" stroke="#38bdf8" stroke-width="2" stroke-opacity="0.75" marker-end="url(#arrow)" />`;
                    }}
                }});

                // Draw Nodes
                Object.keys(nodeCoords).forEach(id => {{
                    const item = nodeCoords[id];
                    const n = item.node;
                    svgHtml += `
                    <g class="concept-node" transform="translate(${{item.x}}, ${{item.y}})" style="cursor: pointer;" onclick="showNodeDetails(${{JSON.stringify(n)}})">
                      <rect width="${{nodeWidth}}" height="${{nodeHeight}}" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="2" style="filter: drop-shadow(0 4px 10px rgba(0,0,0,0.5)); transition: all 0.2s;" />
                      <text x="12" y="24" fill="#38bdf8" font-size="11" font-weight="700">DEPTH ${{n.depth !== undefined ? n.depth : 0}}</text>
                      <text x="12" y="42" fill="#ffffff" font-size="13" font-weight="600" width="176">${{n.label.substring(0, 22)}}</text>
                    </g>
                    `;
                }});

                svgHtml += `</g></svg>`;
                container.innerHTML = svgHtml;

                // Zoom & Pan Handler
                let panX = 0, panY = 0, zoom = 0.85;
                const vp = document.getElementById('svg-viewport');
                function updateTransform() {{
                    vp.setAttribute('transform', `translate(${{panX}}, ${{panY}}) scale(${{zoom}})`);
                }}
                updateTransform();

                return 'SVG RENDERED: ' + data.concept_count + ' NODES';
            }})()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print(r.get('result', {}).get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('svg_rendered_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('result', {}).get('data')))
            print('Saved svg_rendered_screenshot.png')
    finally:
        proc.terminate()

asyncio.run(test_svg())
