import httpx
import os

pptx_path = r"C:\Users\Hiru\Desktop\Research\c1\EduTwin-AI\hiru backend\data\uploads\675bf22f-b1d3-4797-9efc-c525a868f7d5.pptx"
url = "https://bbb.test/edusync/api/c1/lectures/graph"

size_mb = os.path.getsize(pptx_path) / 1024 / 1024
print(f"Uploading {pptx_path} ({size_mb:.2f} MB) to {url}...")

with open(pptx_path, "rb") as f:
    files = {"file": ("lecture.pptx", f, "application/vnd.openxmlformats-officedocument.presentationml.presentation")}
    with httpx.Client(verify=False, timeout=120.0) as client:
        res = client.post(url, files=files)

print("Status Code:", res.status_code)
try:
    data = res.json()
    print("Response JSON summary:")
    print(" - Filename:", data.get("filename"))
    print(" - Page count:", data.get("page_count"))
    print(" - Concept count:", data.get("concept_count"))
    print(" - Concepts:", [c["name"] for c in data.get("concepts", [])[:10]])
    graph = data.get("graph", {})
    print(" - Graph node count:", graph.get("node_count"))
    print(" - Graph edge count:", graph.get("edge_count"))
    print(" - Nodes count:", len(graph.get("nodes", [])))
    print(" - Edges count:", len(graph.get("edges", [])))
    print(" - First 5 nodes:", [n["label"] for n in graph.get("nodes", [])[:5]])
    print(" - First 5 edges:", graph.get("edges", [])[:5])
except Exception as e:
    print("Exception parsing JSON:", e)
    print("Response text:", res.text[:500])
