import requests
import json

def test_api():
    try:
        # Load your data
        with open('worlds.json', 'r', encoding='utf-8') as f:
            tournament_data = json.load(f)
            print(f"Successfully loaded tournament data with {len(tournament_data)} entries")

        # Make the request
        print("Sending request to API...")
        response = requests.post(
            'http://127.0.0.1:8000/analyze',
            json=tournament_data,
            params={'tournament_name': 'Worlds2024'}
        )
        
        # Print response details for debugging
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response Text: {response.text}")
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return
            
        # Try to parse JSON
        try:
            data = response.json()
            print("Successfully parsed JSON response:")
            print(data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            
    except FileNotFoundError:
        print("Could not find tournament data file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_api()