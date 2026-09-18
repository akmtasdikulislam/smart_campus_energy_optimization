import json
import requests

API_URL = "https://smart-campus-energy-optimization-vgd1.onrender.com/optimize-energy"

with open("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for case in data["cases"]:
    case_id = case["id"]
    print(f"Testing {case_id}...")
    
    response = requests.post(API_URL, json=case["input"])
    
    if response.status_code == 200:
        result = response.json()
        expected_cost = case["expected_output"]["total_cost_bdt"]
        actual_cost = result["total_cost_bdt"]
        
        if abs(expected_cost - actual_cost) <= 0.01:
            print(f"✅ Success! Optimal Cost: {actual_cost} BDT")
        else:
            print(f"❌ Failed! Expected Cost: {expected_cost}, Got: {actual_cost}")
    else:
        print(f"⚠️ Error {response.status_code}")