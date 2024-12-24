import os
import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import time

BASE_URL = "https://www.flickr.com/search/?text="  
SEARCH_QUERY = input("Enter the image you want to download(ex., cars,natuer etc):")
DOWNLOAD_FOLDER = "./images"
NUM_IMAGES = 10  

def create_download_folder(folder):
    """Create a folder to save images if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
def scrape_image_urls(base_url, query, max_images=10): #max images ia the number of images you want to download
    """Scrape image URLs from the website."""
    search_url = f"{base_url}{query}"
    response = requests.get(search_url)
    response.raise_for_status()
    soup = bs(response.text, "html.parser")

    
    image_tags = soup.find_all("img", limit=max_images)
    image_urls = []

    for img_tag in image_tags:
        img_url = img_tag.get("src")
        if img_url:
            image_urls.append(urljoin(base_url, img_url))
    return image_urls
def download_images(image_urls, folder):
    #@Sandesh_Yesane
    for i, img_url in enumerate(image_urls, start=1):
        try:
            print(f"Downloading {i}/{len(image_urls)}: {img_url}")
            response = requests.get(img_url)
            response.raise_for_status()
            
            file_name = os.path.join(folder, f"image_{i}.jpg")
            with open(file_name, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")
def main():
    create_download_folder(DOWNLOAD_FOLDER)
    print(f"Scraping image URLs for query: {SEARCH_QUERY}")
    # Scrape image URLs
    image_urls = scrape_image_urls(BASE_URL, SEARCH_QUERY, NUM_IMAGES)
    if not image_urls:
        print("No images found!")
        return

    # Download images
    print(f"Found {len(image_urls)} images. Downloading...")
    download_images(image_urls, DOWNLOAD_FOLDER)
    print("Download complete!")

if __name__ == "__main__":
    time.sleep(2)
    print("Welcome to the image downloader!")
    print("Here we Go!")
    main()
