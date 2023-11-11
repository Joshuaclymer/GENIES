import requests
import tarfile
import os
import fire

def download_tarfile(filename : str, download_dir : str):
    # URL of the file to download
    url = f"https://genies-data.s3.us-east-2.amazonaws.com/{filename}"

    # Specify the local file path where you want to save the downloaded file
    downloaded_file_path = filename

    # Specify the directory where you want to extract the contents
    extracted_dir = download_dir

    # Download the file from the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the downloaded content to a local file
        with open(downloaded_file_path, 'wb') as file:
            file.write(response.content)

        print(f"File '{downloaded_file_path}' has been downloaded successfully.")

        # Create the directory for extraction if it doesn't exist
        os.makedirs(extracted_dir, exist_ok=True)

        # Extract the contents of the tar file
        with tarfile.open(downloaded_file_path, "r") as tar:
            tar.extractall(path=extracted_dir)

        print(f"Contents of '{downloaded_file_path}' have been extracted to '{extracted_dir}'.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def download_data():
    print("Downloading distributions...")
    download_tarfile("genies-datasets.tar", ".")

if __name__ == "__main__":
    fire.Fire(download_data)