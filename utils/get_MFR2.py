# Author: Aqeel Anwar(ICSRL)
# Created: 7/30/2020, 1:44 PM
# Email: aqeel.anwar@gatech.edu

# Code resued from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
import requests,os
from zipfile import ZipFile
import argparse
from urllib2 import urlopen

parser = argparse.ArgumentParser(
    description="Download dataset - Python code to download associated datasets"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="tensorboard",
    help="Name of the dataset - Details on available datasets can be found at GitHub Page",
)
args = parser.parse_args()
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

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

def download(t_url):
    response = urlopen(t_url)
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
    link = 'https://raw.githubusercontent.com/aqeelanwar/PEDRA/master/requirements_cpu.txt'
    links_dict =  Convert(download(link)[0].replace('==', '\n').split('\n'))
    file_id = links_dict[args.dataset]
    file_id = '0B_QQR0eth4z2XzhyNVlDNmptbWM'
    destination = '_.zip'
    download_file_from_google_drive(file_id, destination)
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(destination, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()

    os.remove(destination)

