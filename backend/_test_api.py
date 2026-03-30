import requests

API_KEY = "7f87675b46mshd2da25cdc9eddfcp1fa5a7jsne543a03f7468"
HOST = "politician-trade-tracker1.p.rapidapi.com"

# Let's try to get politicians
url = f"https://{HOST}/get_politicians"
headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": HOST
}

print("Testing /get_politicians endpoint...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    first_key = list(data.keys())[0] if data else None
    if first_key:
        print(f"Sample data keys for {first_key}:", list(data[first_key].keys()))
        print(f"Sample data for {first_key}:", data[first_key])
else:
    print(f"Error {response.status_code}: {response.text}")
