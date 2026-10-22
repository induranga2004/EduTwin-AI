"""Helper script to inject the EduTwin-AI C1 Curriculum Graph drawer and button
directly into BigBlueButton's live HTML5 client (/usr/share/bigbluebutton/html5-client/index.html).
"""

import subprocess
import sys


def main():
    print("[+] Reading /usr/share/bigbluebutton/html5-client/index.html from edutwin-ai-bbb-1...")
    try:
        # Check if backup exists, otherwise create it
        subprocess.run(
            ["docker", "exec", "edutwin-ai-bbb-1", "cp", "-n", "/usr/share/bigbluebutton/html5-client/index.html", "/usr/share/bigbluebutton/html5-client/index.html.orig"],
            check=True,
        )
        res = subprocess.run(
            ["docker", "exec", "edutwin-ai-bbb-1", "cat", "/usr/share/bigbluebutton/html5-client/index.html.orig"],
            capture_output=True,
            text=True,
            check=True,
        )
        html = res.stdout
    except Exception as e:
        print(f"Error reading from container: {e}")
        return 1

    injection = """
<!-- ========================================================================= -->
<!-- EduTwin-AI Component C1 Curriculum Navigation In-Meeting Drawer & Button  -->
<!-- ========================================================================= -->
<style>
  #edutwin-trigger-btn {
    position: fixed;
    bottom: 24px;
    left: 24px;
    z-index: 99998;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    border: 2px solid #38bdf8;
    border-radius: 9999px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: all 0.2s ease;
  }
  #edutwin-trigger-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(56, 189, 248, 0.4);
    filter: brightness(1.1);
  }
  #edutwin-c1-drawer {
    display: none;
    position: fixed;
    top: 0;
    right: 0;
    width: 680px;
    max-width: 90vw;
    height: 100vh;
    background: #0f172a;
    border-left: 2px solid #38bdf8;
    z-index: 99999;
    box-shadow: -15px 0 40px rgba(0, 0, 0, 0.6);
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    animation: edutwinSlideIn 0.25s ease-out;
  }
  @keyframes edutwinSlideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }
  #edutwin-c1-header {
    background: #1e293b;
    padding: 12px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
  }
  #edutwin-c1-iframe {
    flex: 1;
    width: 100%;
    height: calc(100vh - 54px);
    border: none;
    background: #0f172a;
  }
</style>

<button id="edutwin-trigger-btn" onclick="toggleEduTwinDrawer()" title="Toggle Curriculum Intelligence Graph (Shortcut: Alt + C)">
  <span style="font-size: 16px;">🎓</span>
  <span>Curriculum Graph</span>
</button>

<div id="edutwin-c1-drawer">
  <div id="edutwin-c1-header">
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="font-size: 18px;">🎓</span>
      <div>
        <strong style="color: #f8fafc; font-size: 14px; display: block; line-height: 1.2;">EduTwin-AI | Curriculum Intelligence</strong>
        <span style="font-size: 11px; color: #94a3b8;">Lecture Navigation & Prerequisite Tracking (Recording Active)</span>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <a href="/edusync/api/c1/dashboard" target="_blank" style="color: #38bdf8; font-size: 12px; text-decoration: none; border: 1px solid rgba(56,189,248,0.3); padding: 5px 10px; border-radius: 6px; background: rgba(56,189,248,0.1); font-weight: 600;">↗️ Open in New Window</a>
      <button onclick="toggleEduTwinDrawer()" style="background: none; border: none; color: #94a3b8; font-size: 20px; cursor: pointer; padding: 4px 8px; border-radius: 4px; line-height: 1;" title="Close Panel">✕</button>
    </div>
  </div>
  <iframe id="edutwin-c1-iframe" src="/edusync/api/c1/dashboard"></iframe>
</div>

<script>
  function toggleEduTwinDrawer() {
    var drawer = document.getElementById('edutwin-c1-drawer');
    if (drawer) {
      if (drawer.style.display === 'none' || drawer.style.display === '') {
        drawer.style.display = 'flex';
      } else {
        drawer.style.display = 'none';
      }
    }
  }

  // Keyboard shortcut: Alt + C
  document.addEventListener('keydown', function(e) {
    if (e.altKey && (e.key === 'c' || e.key === 'C')) {
      e.preventDefault();
      toggleEduTwinDrawer();
    }
  });
</script>
</body>
"""

    if "</body>" not in html:
        print("Error: </body> tag not found in index.html")
        return 1

    new_html = html.replace("</body>", injection)

    # Pipe directly into container
    proc = subprocess.Popen(
        ["docker", "exec", "-i", "edutwin-ai-bbb-1", "tee", "/usr/share/bigbluebutton/html5-client/index.html"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    proc.communicate(input=new_html)
    print("[+] Successfully injected Curriculum Graph into BigBlueButton HTML5 client!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
