import asyncio, json, urllib.request, websockets, subprocess, httpx, base64

async def test_live_conversation():
    print("[+] Loading lecture graph for user1.pdf...")
    with open(r'hiru backend\data\uploads\8b0315c3-b1de-48b4-b2cd-70a02ac350d8.pdf', 'rb') as f:
        files = {'file': ('user1.pdf', f, 'application/pdf')}
        with httpx.Client(verify=False, timeout=60.0) as client:
            res = client.post('https://bbb.test/edusync/api/c1/lectures/graph', files=files)
    backend_data = res.json()

    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--remote-debugging-port=9997',
        '--headless=new',
        '--ignore-certificate-errors',
        '--window-size=1920,1080'
    ])
    await asyncio.sleep(1.8)
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
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2.5)

            # 1. Render Graph Data in Dashboard
            js_render = f"""
            (() => {{
                const data = {json.dumps(backend_data)};
                displayGraphData(data, "Database Security with Oracle (.pdf)");
                return {{
                    totalNodes: document.querySelectorAll('.concept-node').length,
                    rootTitle: document.getElementById('root-title').textContent
                }};
            }})()
            """
            r1 = await call('Runtime.evaluate', {'expression': js_render, 'returnByValue': True})
            print("[+] Rendered Graph:", r1.get('result', {}).get('result', {}).get('value'))
            await asyncio.sleep(1)

            # 2. Simulate Spoken Conversation Utterances (Live Speech Stream)
            utterances = [
                ("Lecturer (Live)", "Welcome everyone. In this live session we begin with Database Security Fundamentals and enterprise compliance."),
                ("Lecturer (Live)", "Now let's examine Common Threats to Databases, specifically SQL Injection and unauthorized access."),
                ("Student (Question)", "Professor, could you explain the Core Principles of Database Security like Confidentiality and Integrity?"),
                ("Lecturer (Answer)", "Certainly, confidentiality ensures only authorized users read data, while integrity prevents unauthorized tampering."),
                ("Lecturer (Jump)", "Let's fast-forward directly to Oracle User Authentication and multifactor authentication protocols.")
            ]

            for speaker, text in utterances:
                js_utterance = f"""
                (() => {{
                    processLiveConversationUtterance({json.dumps(text)}, {json.dumps(speaker)});
                }})()
                """
                await call('Runtime.evaluate', {'expression': js_utterance})
                await asyncio.sleep(0.5)

            # 3. Check Live Transcript Feed DOM state
            js_inspect = """
            (() => {
                const transcriptItems = Array.from(document.querySelectorAll('.transcript-item')).map(item => ({
                    speaker: item.querySelector('.transcript-meta strong') ? item.querySelector('.transcript-meta strong').textContent : '',
                    time: item.querySelector('.transcript-meta span:last-child') ? item.querySelector('.transcript-meta span:last-child').textContent : '',
                    hasHighlight: item.querySelectorAll('.concept-hl').length > 0,
                    hasConceptTag: item.querySelectorAll('.concept-tag').length > 0,
                    conceptTagText: item.querySelector('.concept-tag') ? item.querySelector('.concept-tag').textContent.replace(/\\s+/g, ' ').trim() : ''
                }));
                const skippedCount = Array.from(document.querySelectorAll('.concept-node rect')).filter(r => r.getAttribute('stroke') === '#ef4444').length;
                const coveredCount = Array.from(document.querySelectorAll('.concept-node rect')).filter(r => r.getAttribute('stroke') === '#22c55e').length;
                const alertMsg = document.getElementById('skip-alert-msg') ? document.getElementById('skip-alert-msg').textContent : '';

                return {
                    transcriptCount: transcriptItems.length,
                    items: transcriptItems,
                    skippedCardsOnGraph: skippedCount,
                    coveredCardsOnGraph: coveredCount,
                    skipAlertVisible: document.getElementById('skip-alert-box').style.display !== 'none',
                    skipAlertMessage: alertMsg
                };
            })()
            """
            r2 = await call('Runtime.evaluate', {'expression': js_inspect, 'returnByValue': True})
            print("[+] Live Conversation & Transcript Status:\n", json.dumps(r2.get('result', {}).get('result', {}).get('value'), indent=2))
            await asyncio.sleep(1)

            # Capture screenshot
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            data_b64 = shot.get('result', {}).get('data', '')
            out_file = 'verified_live_conversation_transcript_graph.png'
            with open(out_file, 'wb') as f:
                f.write(base64.b64decode(data_b64))
            print(f"[+] Saved live conversation screenshot successfully: {out_file}!")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == '__main__':
    asyncio.run(test_live_conversation())
