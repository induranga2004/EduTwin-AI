import asyncio, json, urllib.request, websockets, subprocess, base64

async def check():
    proc = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--remote-debugging-port=9999', '--headless=new', '--ignore-certificate-errors', '--window-size=1920,1080'])
    await asyncio.sleep(1)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9999/json')
        ws_url = [t for t in json.loads(resp.read().decode()) if t.get('type') == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url) as ws:
            async def call(m, p={}):
                await ws.send(json.dumps({'id': 1, 'method': m, 'params': p}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get('id') == 1: return d.get('result', {})
            await call('Page.navigate', {'url': 'https://bbb.test/edusync/api/c1/dashboard'})
            await asyncio.sleep(2)
            
            js = """
            (() => {
                const container = document.getElementById('graph-canvas');
                
                // Let's create the most basic vis.Network possible
                const data = {
                    nodes: new vis.DataSet([
                        { id: 1, label: 'HELLO WORLD', shape: 'box', color: '#ff0000', size: 30 }
                    ]),
                    edges: new vis.DataSet([])
                };
                const net = new vis.Network(container, data, {
                    physics: false
                });
                net.fit();

                const canvas = container.querySelector('canvas');
                const ctx = canvas.getContext('2d');
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                // Count non-black / non-transparent pixels
                let nonZero = 0;
                for (let i = 0; i < imgData.data.length; i += 4) {
                    if (imgData.data[i] !== 0 || imgData.data[i+1] !== 0 || imgData.data[i+2] !== 0) {
                        nonZero++;
                    }
                }
                
                return {
                    canvasWidth: canvas.width,
                    canvasHeight: canvas.height,
                    nonZeroPixels: nonZero,
                    nodePositions: net.getPositions()
                };
            })()
            """
            r = await call('Runtime.evaluate', {'expression': js, 'returnByValue': True})
            print("RESULT:", r.get('result', {}).get('value'))
            await asyncio.sleep(1)
            
            shot = await call('Page.captureScreenshot', {'format': 'png'})
            with open('test_pixel_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(shot.get('data')))
            print("Saved test_pixel_screenshot.png")
    finally:
        proc.terminate()

asyncio.run(check())
