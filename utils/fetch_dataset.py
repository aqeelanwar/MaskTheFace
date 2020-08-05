# Author: Aqeel Anwar(ICSRL)
# Created: 7/30/2020, 1:44 PM
# Email: aqeel.anwar@gatech.edu

# Code resued from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
# Make sure you run this from parent folder and not from utils folder i.e.
# python utils/fetch_dataset.py

import requests, os
from zipfile import ZipFile
import argparse
import urllib

parser = argparse.ArgumentParser(
    description="Download dataset - Python code to download associated datasets"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="mfr2",
    help="Name of the dataset - Details on available datasets can be found at GitHub Page",
)
args = parser.parse_args()


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    print(destination)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download(t_url):
    response = urllib.request.urlopen(t_url)
    data = response.read()
    txt_str = str(data)
    lines = txt_str.split("\\n")
    return lines


def Convert(lst):
    it = iter(lst)
    res_dct = dict(zip(it, it))
    return res_dct


if __name__ == "__main__":
    # Fetch the latest download_links.txt file from GitHub
    link = "https://raw.githubusercontent.com/aqeelanwar/MaskTheFace/master/datasets/download_links.txt"
    links_dict = Convert(
        download(link)[0]
        .replace(":", "\n")
        .replace("b'", "")
        .replace("'", "")
        .replace(" ", "")
        .split("\n")
    )
    file_id = links_dict[args.dataset]
    destination = "datasets/_.zip"
    print("Downloading: ", args.dataset)
    download_file_from_google_drive(file_id, destination)
    print("Extracting: ", args.dataset)
    with ZipFile(destination, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(destination.rsplit(os.path.sep, 1)[0])

    os.remove(destination)
