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
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    z-index: 2147483647 !important;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 9999px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    box-shadow: 0 4px 25px rgba(2, 132, 199, 0.6) !important;
    transition: all 0.2s ease !important;
    animation: edutwinPulse 3s infinite !important;
  }
  @keyframes edutwinPulse {
    0%, 100% { box-shadow: 0 4px 25px rgba(2, 132, 199, 0.6); }
    50% { box-shadow: 0 6px 35px rgba(56, 189, 248, 0.95); transform: translateY(-2px); }
  }
  #edutwin-trigger-btn:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-3px) !important;
  }
  #edutwin-c1-drawer {
    display: none;
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    width: 860px !important;
    max-width: 95vw !important;
    height: 100vh !important;
    background: #0f172a !important;
    border-left: 2px solid #38bdf8 !important;
    z-index: 2147483647 !important;
    box-shadow: -15px 0 40px rgba(0, 0, 0, 0.7) !important;
    flex-direction: column !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    animation: edutwinSlideIn 0.25s ease-out !important;
  }
  @keyframes edutwinSlideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }
  #edutwin-c1-header {
    background: #1e293b !important;
    padding: 12px 18px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    border-bottom: 1px solid #334155 !important;
  }
  #edutwin-c1-iframe {
    flex: 1 !important;
    width: 100% !important;
    height: calc(100vh - 54px) !important;
    border: none !important;
    background: #0f172a !important;
  }
</style>

<button id="edutwin-trigger-btn" onclick="toggleEduTwinDrawer()" title="Toggle Curriculum Intelligence Graph (Shortcut: Alt + C)">
  <span style="font-size: 18px;">🎓</span>
  <span>Curriculum Graph</span>
</button>

<div id="edutwin-c1-drawer">
  <div id="edutwin-c1-header">
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="font-size: 20px;">🎓</span>
      <div>
        <strong style="color: #f8fafc; font-size: 14px; display: block; line-height: 1.2;">EduTwin-AI | Curriculum Intelligence</strong>
        <span style="font-size: 11px; color: #94a3b8;">Lecture Navigation & Prerequisite Tracking (Recording Active)</span>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <a href="/edusync/api/c1/dashboard" target="_blank" style="color: #38bdf8; font-size: 12px; text-decoration: none; border: 1px solid rgba(56,189,248,0.3); padding: 5px 10px; border-radius: 6px; background: rgba(56,189,248,0.1); font-weight: 600;">↗️ Open in New Window</a>
      <button onclick="toggleEduTwinDrawer()" style="background: none; border: none; color: #94a3b8; font-size: 22px; cursor: pointer; padding: 4px 8px; border-radius: 4px; line-height: 1;" title="Close Panel">✕</button>
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

  // Ensure elements remain attached even if React replaces body children
  setInterval(function() {
    var btn = document.getElementById('edutwin-trigger-btn');
    var drawer = document.getElementById('edutwin-c1-drawer');
    if (btn && btn.parentElement !== document.body) {
      document.body.appendChild(btn);
    }
    if (drawer && drawer.parentElement !== document.body) {
      document.body.appendChild(drawer);
    }
  }, 1000);
</script>
</body>
"""

    if "</body>" not in html:
        print("Error: </body> tag not found in index.html")
        return 1

    new_html = html.replace("</body>", injection)

    # Pipe directly into container using utf-8 encoding
    proc = subprocess.Popen(
        ["docker", "exec", "-i", "edutwin-ai-bbb-1", "tee", "/usr/share/bigbluebutton/html5-client/index.html"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    proc.communicate(input=new_html)

    # GZIP compress index.html so Nginx gzip_static on serves the updated file
    subprocess.run(
        ["docker", "exec", "edutwin-ai-bbb-1", "gzip", "-k", "-f", "/usr/share/bigbluebutton/html5-client/index.html"],
        check=True,
    )
    print("[+] Successfully injected Curriculum Graph into BigBlueButton HTML5 client and gzipped for Nginx!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
