import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_subtopics_and_notifications():
    print("[+] Fetching demo graph for Cyber_Resilience_Energy_Sector.pptx...")
    with httpx.Client(verify=False, timeout=60.0) as client:
        res = client.post('https://bbb.test/edusync/api/c1/lectures/demo-graph?name=Cyber_Resilience_Energy_Sector.pptx')
    backend_data = res.json()
    print(f"[+] Retrieved graph with {len(backend_data['graph']['nodes'])} nodes and {len(backend_data['graph']['edges'])} edges.")

    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--remote-debugging-port=9997',
        '--headless=new',
        '--ignore-certificate-errors',
        '--window-size=1920,1080'
    ])
    await asyncio.sleep(2.0)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9997/json')
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

            # Check if there are any errors by evaluating console logs in JS
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2.5)

            # 1. Render Graph Data in Dashboard
            js_render = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                displayGraphData(data, "Cyber_Resilience_Energy_Sector.pptx");
                return {{
                    totalNodes: document.querySelectorAll('.concept-node').length,
                    mainTopicCards: document.querySelectorAll('.main-topic-node').length,
                    subTopicCards: document.querySelectorAll('.subtopic-node').length,
                    rootTitle: document.getElementById('root-title') ? document.getElementById('root-title').textContent : ''
                }};
            }})()
            """
            r1 = await call('Runtime.evaluate', {'expression': js_render, 'returnByValue': True})
            print("[+] Rendered Subtopics & Main Topics:", r1.get('result', {}).get('result', {}).get('value'))
            await asyncio.sleep(1.0)

            # 2. Spoken Utterance 1: Deliver Sub-Topic under Slide 02
            print("[+] Simulating Lecturer covering Sub-Topic 1 under Slide 02...")
            u1 = "resilience extends cybersecurity from protection alone to continuity adaptation and recovery"
            js_u1 = f"""
            (async () => {{
                await processLiveConversationUtterance({json.dumps(u1)}, "Lecturer (Live Mic)");
            }})()
            """
            await call('Runtime.evaluate', {'expression': js_u1, 'awaitPromise': True})
            await asyncio.sleep(1.5)

            # 3. Spoken Utterance 2: Deliver Slide 03
            print("[+] Simulating Lecturer delivering Slide 03...")
            u2 = "Now moving to Research Statement and Objectives identifying key research questions on energy sector cybersecurity"
            js_u2 = f"""
            (async () => {{
                await processLiveConversationUtterance({json.dumps(u2)}, "Lecturer (Live Mic)");
            }})()
            """
            await call('Runtime.evaluate', {'expression': js_u2, 'awaitPromise': True})
            await asyncio.sleep(1.5)

            # 4. Spoken Utterance 3: SKIP VIOLATION! Jump directly to Slide 07: Resilience Strategies
            print("[+] Simulating Lecturer skipping directly to Slide 07 (Resilience Strategies)...")
            u3 = "Let us jump directly to Resilience Strategies and framework recommendations for smart grids"
            js_u3 = f"""
            (async () => {{
                await processLiveConversationUtterance({json.dumps(u3)}, "Lecturer (Live Skip Jump)");
            }})()
            """
            await call('Runtime.evaluate', {'expression': js_u3, 'awaitPromise': True})
            await asyncio.sleep(1.5)

            # 5. Inspect UI State (Notifications, Graph Colors, Counters)
            js_inspect = """
            (() => {
                const subNodes = Array.from(document.querySelectorAll('.subtopic-node')).map(el => ({
                    id: el.dataset.nodeId,
                    parentId: el.dataset.parentId,
                    title: el.querySelector('text[id^="title-sub"]') ? el.querySelector('text[id^="title-sub"]').textContent : '',
                    badge: el.querySelector('text[id^="badge-sub"]') ? el.querySelector('text[id^="badge-sub"]').textContent : '',
                    stroke: el.querySelector('rect') ? el.querySelector('rect').getAttribute('stroke') : ''
                }));

                const mainNodes = Array.from(document.querySelectorAll('.main-topic-node')).map(el => ({
                    id: el.dataset.nodeId,
                    title: el.querySelector('text[id^="title-"]') ? el.querySelector('text[id^="title-"]').textContent : '',
                    badge: el.querySelector('text[id^="badge-"]') ? el.querySelector('text[id^="badge-"]').textContent : '',
                    subCounter: el.querySelector('text[id^="sub-counter-"]') ? el.querySelector('text[id^="sub-counter-"]').textContent : '',
                    stroke: el.querySelector('rect') ? el.querySelector('rect').getAttribute('stroke') : ''
                }));

                const liveBanner = document.getElementById('live-notification-banner');
                const floatingBanner = document.getElementById('graph-floating-notification');
                const skipAlertBox = document.getElementById('skip-alert-box');

                return {
                    deliveredSubtopicIds: Array.from(deliveredSubtopicIds),
                    deliveredConceptIds: Array.from(deliveredConceptIds),
                    skippedConceptIds: Array.from(skippedConceptIds),
                    liveBannerHtml: liveBanner ? liveBanner.innerText : '',
                    floatingBannerText: floatingBanner ? floatingBanner.innerText : '',
                    skipAlertVisible: skipAlertBox ? skipAlertBox.style.display !== 'none' : false,
                    skipAlertMsg: document.getElementById('skip-alert-msg') ? document.getElementById('skip-alert-msg').innerText : '',
                    totalMainNodes: mainNodes.length,
                    totalSubNodes: subNodes.length,
                    subNodeDetails: subNodes.slice(0, 5),
                    coveredMainNodes: mainNodes.filter(n => n.badge.includes('COVERED')).map(n => n.title),
                    skippedMainNodes: mainNodes.filter(n => n.badge.includes('SKIPPED')).map(n => n.title),
                    coveredSubNodes: subNodes.filter(n => n.badge.includes('COVERED')).map(n => n.title),
                    skippedSubNodes: subNodes.filter(n => n.badge.includes('SKIPPED')).map(n => n.title)
                };
            })()
            """
            r_inspect = await call('Runtime.evaluate', {'expression': js_inspect, 'returnByValue': True})
            ui_state = r_inspect.get('result', {}).get('result', {}).get('value', {})
            print("\n[+] Verification UI State:\n", json.dumps(ui_state, indent=2))

            # 6. Capture Screenshot
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            data_b64 = shot.get('result', {}).get('data', '')
            out_file = 'verified_subtopics_and_notifications_live_graph.png'
            with open(out_file, 'wb') as f:
                f.write(base64.b64decode(data_b64))
            print(f"\n[+] Successfully saved screenshot: {out_file}!")

            # Copy to artifacts directory
            import shutil
            artifact_dest = r'C:\Users\Hiru\.gemini\antigravity-cli\brain\85d9fbdb-b8b0-4da8-848d-5fdbc2b51bc3\verified_subtopics_and_notifications_live_graph.png'
            shutil.copyfile(out_file, artifact_dest)
            print(f"[+] Copied screenshot to artifacts directory: {artifact_dest}")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    asyncio.run(test_subtopics_and_notifications())
