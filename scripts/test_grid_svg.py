import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_grid_svg():
    with open(r'hiru backend\data\uploads\8b0315c3-b1de-48b4-b2cd-70a02ac350d8.pdf', 'rb') as f:
        files = {'file': ('lecture.pdf', f, 'application/pdf')}
        with httpx.Client(verify=False, timeout=60.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

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
            
            js = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                const container = document.getElementById('graph-canvas');
                document.getElementById('canvas-placeholder').style.display = 'none';
                document.getElementById('canvas-toolbar').style.display = 'flex';
                document.getElementById('course-label').innerText = data.concept_count + ' Concepts Extracted';
                document.getElementById('graph-stats').innerText = data.concept_count + ' Nodes | ' + data.graph.edges.length + ' Prerequisite Edges';

                const cardW = 190;
                const cardH = 58;
                const cols = 5;
                const gapX = 35;
                const gapY = 55;
                const padX = 40;
                const padY = 40;

                const nodeCoords = {{}};
                data.graph.nodes.forEach((n, idx) => {{
                    const col = idx % cols;
                    const row = Math.floor(idx / cols);
                    const x = padX + col * (cardW + gapX);
                    const y = padY + row * (cardH + gapY);
                    nodeCoords[n.id] = {{ x, y, n }};
                }});

                let svg = `
                <svg id="svg-root" width="100%" height="100%" style="background: #090d16; cursor: grab; user-select: none;">
                  <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#38bdf8" />
                    </marker>
                    <linearGradient id="nodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#1e293b" />
                      <stop offset="100%" stop-color="#0f172a" />
                    </linearGradient>
                    <linearGradient id="activeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#065f46" />
                      <stop offset="100%" stop-color="#022c22" />
                    </linearGradient>
                  </defs>
                  <g id="viewport" transform="translate(20, 20) scale(0.95)">
                `;

                // Edges
                data.graph.edges.forEach(e => {{
                    const s = nodeCoords[e.source];
                    const t = nodeCoords[e.target];
                    if (s && t) {{
                        const x1 = s.x + cardW / 2;
                        const y1 = s.y + cardH;
                        const x2 = t.x + cardW / 2;
                        const y2 = t.y;
                        const my = (y1 + y2) / 2;
                        svg += `<path d="M ${{x1}} ${{y1}} C ${{x1}} ${{my}}, ${{x2}} ${{my}}, ${{x2}} ${{y2}}" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-opacity="0.65" marker-end="url(#arrow)" />`;
                    }}
                }});

                // Nodes
                data.graph.nodes.forEach(n => {{
                    const item = nodeCoords[n.id];
                    const label = n.label.replace('\\n', ' ');
                    svg += `
                    <g class="node-item" id="node-${{n.id}}" transform="translate(${{item.x}}, ${{item.y}})" style="cursor: pointer;">
                      <rect width="${{cardW}}" height="${{cardH}}" rx="8" fill="url(#nodeGrad)" stroke="#38bdf8" stroke-width="1.5" style="filter: drop-shadow(0 4px 12px rgba(0,0,0,0.6));" />
                      <circle cx="16" cy="18" r="4" fill="#38bdf8" />
                      <text x="26" y="22" fill="#38bdf8" font-size="10" font-weight="700" letter-spacing="0.5">LEVEL ${{n.depth || 1}}</text>
                      <text x="14" y="42" fill="#f8fafc" font-size="12" font-weight="600">${{label.substring(0, 20)}}</text>
                    </g>
                    `;
                }});

                svg += `</g></svg>`;
                container.innerHTML = svg;
                return 'GRID RENDERED: ' + data.concept_count;
            }})()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print(r.get('result', {}).get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('grid_svg_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('result', {}).get('data')))
            print('Saved grid_svg_screenshot.png')
    finally:
        proc.terminate()

asyncio.run(test_grid_svg())
