# Contract test script — pehle server chalayein
#   (python -m uvicorn main:app --port 8000), phir ye chalain:
#   python test_api.py <page-image>
# Misal:  python test_api.py gold_test_page.jpg
#
# Flow (frontend ke contract ke mutabiq):
#   register -> login (JSON, token) -> detection task -> poll (har 1s,
#   asal frontend 2.5s karta hai) jab tak completed/failed na ho ->
#   detection history -> verification task -> poll -> logout ->
#   purane token se /me (401 hona chahiye)

import sys
import time

import requests

API = "http://127.0.0.1:8000"
BASE = f"{API}/api/v1"
EMAIL = "test@example.com"
PASSWORD = "password123"


def poll(url, headers, label):
    for _ in range(60):
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("completed", "failed"):
            print(f"   {label} final status:", data["status"])
            return data
        time.sleep(1)
    print(f"   {label}: 60s me final status nahi aya")
    return data


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gold_test_page.jpg"

    r = requests.post(
        f"{BASE}/auth/register",
        json={"email": EMAIL, "password": PASSWORD, "full_name": "Test User"},
    )
    print("1) Register — Status:", r.status_code)

    r = requests.post(f"{BASE}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    r.raise_for_status()
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("2) Login OK — user:", r.json().get("user"))

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/signature-detection", files={"document": f}, headers=headers
        )
    r.raise_for_status()
    det_id = r.json()["task_id"]
    print("3) Detection task bana:", det_id)
    det = poll(f"{BASE}/signature-detection/{det_id}", headers, "Detection")
    if det["status"] == "completed":
        res = det["result"]
        print("   signature_count:", res["signature_count"])
        for d in res["detections"]:
            print("   -", d)
    else:
        print("   error:", det.get("error"))

    hist = requests.get(f"{BASE}/signature-detection", headers=headers).json()
    print("4) Detection history — total:", hist["total"], "| items:", len(hist["items"]))

    with open(path, "rb") as f1, open(path, "rb") as f2:
        r = requests.post(
            f"{BASE}/signature-verification",
            files={"original_signature": f1, "suspected_signature": f2},
            headers=headers,
        )
    r.raise_for_status()
    ver_id = r.json()["task_id"]
    print("5) Verification task bana:", ver_id)
    ver = poll(f"{BASE}/signature-verification/{ver_id}", headers, "Verification")
    print("   (weights file na ho to expected: failed — VERIFICATION_MODEL_NOT_CONNECTED;")
    print("    siamese.weights.h5 + TensorFlow ho to expected: completed, similarity_score/verdict ke saath)")
    print("   error:", ver.get("error"))

    out = requests.post(f"{BASE}/auth/logout", headers=headers)
    again = requests.get(f"{BASE}/auth/me", headers=headers)
    print("6) Logout — Status:", out.status_code,
          "| purane token se /me — Status:", again.status_code, "(401 hona chahiye)")


if __name__ == "__main__":
    main()
