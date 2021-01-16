import requests

FILE_ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"

def download_file_from_google_drive(destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : FILE_ID }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : FILE_ID, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 2:
        print("Usage: python download_celeba.py destination_file_path")
    else:
        destination = sys.argv[1]
        download_file_from_google_drive(destination)