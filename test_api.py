import requests

def test_get_final_answer():
    url = "http://127.0.0.1:8000/get-final-answer"
    payload = {"task": "Describe the hair type based on the image."}
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    print("Testing /get-final-answer endpoint...")
    test_get_final_answer()