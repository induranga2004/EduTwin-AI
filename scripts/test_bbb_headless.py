import subprocess
import time
import os

url = "https://bbb.test/bigbluebutton/api/join?fullName=Lecturer&meetingID=edusync-demo&password=mp&redirect=true&checksum=e00bd472b06210a72def534ffd1fb51d68ac53d9"
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
screenshot_path = r"C:\Users\Hiru\Desktop\Research\c1\EduTwin-AI\bbb_screenshot.png"
dom_path = r"C:\Users\Hiru\Desktop\Research\c1\EduTwin-AI\bbb_dom.html"

cmd = [
    chrome_path,
    "--headless=new",
    "--ignore-certificate-errors",
    "--window-size=1920,1080",
    "--virtual-time-budget=10000",
    f"--screenshot={screenshot_path}",
    f"--dump-dom",
    url
]

print("Launching Chrome...")
res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
print("Return code:", res.returncode)
with open(dom_path, "w", encoding="utf-8") as f:
    f.write(res.stdout)

print("Saved dom to", dom_path, "bytes:", len(res.stdout))
if os.path.exists(screenshot_path):
    print("Screenshot exists! Size:", os.path.getsize(screenshot_path))
else:
    print("No screenshot generated")
