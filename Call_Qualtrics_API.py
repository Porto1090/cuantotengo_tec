import requests, json
from datetime import datetime

QUALTRICS_API_TOKEN = "YOUR_API_TOKEN"
QUALTRICS_DATA_CENTER = "iad1"
QUALTRICS_SURVEY_ID = "SV_XXXXXXX"

def create_prefilled_response(user_id, phase, product_summary):
    url = f"https://{QUALTRICS_DATA_CENTER}.qualtrics.com/API/v3/responseimports/{QUALTRICS_SURVEY_ID}"
    headers = {
        "X-API-TOKEN": QUALTRICS_API_TOKEN,
        "Content-Type": "application/json"
    }

    payload = {
        "format": "json",
        "responses": [{
            "values": {
                "UserID": user_id,
                "PhaseNumber": phase,
                "ProductSummary": product_summary,
                "timestamp": datetime.utcnow().isoformat()
            }
        }]
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        print("❌ Upload failed:", r.text)
        return None
    data = r.json()
    response_id = data["result"]["importId"]  # Qualtrics returns an importId for this upload
    print("✅ Pre-filled response created:", response_id)
    return response_id
