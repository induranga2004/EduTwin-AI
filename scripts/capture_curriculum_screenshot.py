import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_svg():
    print("[+] Requesting graph for user1.pdf...")
    with open(r'hiru backend\data\uploads\8b0315c3-b1de-48b4-b2cd-70a02ac350d8.pdf', 'rb') as f:
        files = {'file': ('user1.pdf', f, 'application/pdf')}
        with httpx.Client(verify=False, timeout=60.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()
    graph = backend_data.get('graph', {})
    print(f"[+] Loaded: {graph.get('node_count')} nodes, {graph.get('edge_count')} edges")
    print(f"[+] Main concept: {graph.get('main_concept')}")

    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--remote-debugging-port=9998',
        '--headless=new',
        '--ignore-certificate-errors',
        '--window-size=1920,1080'
    ])
    await asyncio.sleep(1.8)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9998/json')
        ws_url = [t for t in json.loads(resp.read().decode()) if t.get('type') == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url, max_size=30*1024*1024) as ws:
            msg_counter = 1
            async def call(m, p={}):
                nonlocal msg_counter
                cid = msg_counter
                msg_counter += 1
                await ws.send(json.dumps({'id': cid, 'method': m, 'params': p}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get('id') == cid:
                        return d
            await call('Runtime.enable')
            await call('Page.enable')
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2.5)

            # 1. Render Graph Data in Dashboard
            js_render = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                displayGraphData(data, "Database Security with Oracle (.pdf)");
                return {{
                    totalNodes: document.querySelectorAll('.concept-node').length,
                    rootTitle: document.getElementById('root-title') ? document.getElementById('root-title').textContent : '',
                    rootStatus: document.getElementById('root-status-pill') ? document.getElementById('root-status-pill').textContent : ''
                }};
            }})()
            """
            r1 = await call('Runtime.evaluate', {'expression': js_render, 'returnByValue': True})
            print("[+] Render State:", json.dumps(r1.get('result', {}).get('result', {}).get('value'), indent=2))
            await asyncio.sleep(1)

            # 2. Advance Slide 02 Sequentially
            js_step1 = """
            (() => {
                deliverNextSequentialSlide();
                return {
                    transcript: document.getElementById('transcript-ticker').textContent,
                    stats: document.getElementById('coverage-stats-label').textContent,
                    alertVisible: document.getElementById('skip-alert-box').style.display !== 'none'
                };
            })()
            """
            r2 = await call('Runtime.evaluate', {'expression': js_step1, 'returnByValue': True})
            print("[+] Step 1 (Sequential Delivery):", json.dumps(r2.get('result', {}).get('result', {}).get('value'), indent=2))
            await asyncio.sleep(1)

            # 3. Simulate Concept Skip Violation: Jump directly to Slide 07!
            js_skip = """
            (() => {
                simulatePrerequisiteSkip();
                const alertMsg = document.getElementById('skip-alert-msg') ? document.getElementById('skip-alert-msg').textContent : '';
                const skippedCards = Array.from(document.querySelectorAll('.concept-node rect')).filter(r => r.getAttribute('stroke') === '#ef4444');
                return {
                    alertVisible: document.getElementById('skip-alert-box').style.display !== 'none',
                    alertMessage: alertMsg,
                    redDashedSkippedCount: skippedCards.length,
                    stats: document.getElementById('coverage-stats-label').textContent
                };
            })()
            """
            r3 = await call('Runtime.evaluate', {'expression': js_skip, 'returnByValue': True})
            print("[+] Step 2 (Prerequisite Skip Alert):", json.dumps(r3.get('result', {}).get('result', {}).get('value'), indent=2))
            await asyncio.sleep(1)

            # 4. Advance through remaining slides to fulfill constituent content and trigger Main Concept synthesized coverage!
            js_synth = """
            (() => {
                for (let i = 0; i < 7; i++) {
                    deliverNextSequentialSlide();
                }
                const rootStatus = document.getElementById('root-status-pill').textContent;
                const rootStroke = document.getElementById('rect-root').getAttribute('stroke');
                const synthLabel = document.getElementById('main-concept-synthesis-label').textContent;
                return {
                    rootStatus,
                    rootStroke,
                    synthLabel,
                    stats: document.getElementById('coverage-stats-label').textContent
                };
            })()
            """
            r4 = await call('Runtime.evaluate', {'expression': js_synth, 'returnByValue': True})
            print("[+] Step 3 (Main Topic Synthesized Coverage):", json.dumps(r4.get('result', {}).get('result', {}).get('value'), indent=2))
            await asyncio.sleep(1)

            # Capture screenshot
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            data_b64 = shot.get('result', {}).get('data', '')
            out_file = 'verified_curriculum_graph.png'
            with open(out_file, 'wb') as f:
                f.write(base64.b64decode(data_b64))
            print(f"[+] Saved screenshot successfully: {out_file}!")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    asyncio.run(test_svg())
