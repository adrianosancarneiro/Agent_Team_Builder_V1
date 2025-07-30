import requests

class WebAPITool:
    """Tool to make external web API calls (internet access)."""
    def get(self, url: str, params: dict = None, headers: dict = None) -> str:
        """Perform a GET request to the given URL and return response text."""
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            return res.text
        except Exception as e:
            return f"ERROR: {e}"

    def post(self, url: str, data: dict = None, json_data: dict = None, headers: dict = None) -> str:
        """Perform a POST request (with JSON or form data) and return response text."""
        try:
            res = requests.post(url, data=data, json=json_data, headers=headers, timeout=10)
            res.raise_for_status()
            return res.text
        except Exception as e:
            return f"ERROR: {e}"
