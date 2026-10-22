import asyncio, json, urllib.request, websockets, subprocess, base64

async def debug():
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
            
            js = """
            (() => {
                const container = document.getElementById('graph-canvas');
                container.innerHTML = '<canvas id="direct-canvas" width="1400" height="800" style="width:100%; height:100%; background: #112233;"></canvas>';
                const cvs = document.getElementById('direct-canvas');
                const ctx = cvs.getContext('2d');
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(100, 100, 300, 200);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 24px Arial';
                ctx.fillText('DIRECT CANVAS TEST', 120, 210);
                document.getElementById('canvas-placeholder').style.display = 'none';
                return 'DRAWN DIRECTLY';
            })()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print(r.get('result', {}).get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('direct_canvas_test.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('result', {}).get('data')))
            print('Saved direct_canvas_test.png')
    finally:
        proc.terminate()

asyncio.run(debug())
