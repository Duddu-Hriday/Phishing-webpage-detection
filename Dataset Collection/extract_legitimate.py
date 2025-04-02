import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, urljoin
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

# Function to compute the hash of HTML content
def compute_hash(html_content):
    return hashlib.md5(html_content.encode('utf-8')).hexdigest()

# Check if a page is a duplicate based on its hash
def is_duplicate(folder_path, content_hash):
    hash_file = os.path.join(folder_path, 'hashes.txt')

    if os.path.exists(hash_file):
        with open(hash_file, 'r', encoding='utf-8') as file:
            existing_hashes = file.readlines()
            if content_hash + '\n' in existing_hashes:
                return True
    return False

# Function to save the page's hash to avoid duplicates
def save_hash(folder_path, content_hash):
    hash_file = os.path.join(folder_path, 'hashes.txt')
    with open(hash_file, 'a', encoding='utf-8') as file:
        file.write(content_hash + '\n')

# Function to save metadata
def save_metadata(url, folder_path, title, num_links, num_images, contains_forms):
    metadata = {
        "URL": url,
        "Title": title,
        "Num_Links": num_links,
        "Num_Images": num_images,
        "Contains_Forms": contains_forms
    }

    metadata_path = os.path.join(folder_path, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

# Function to download resources (CSS/JS) with dynamic extension
def download_file(file_url, folder_path, expected_type):
    try:
        file_response = requests.get(file_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        file_response.raise_for_status()

        content_type = file_response.headers.get('Content-Type', '').lower()

        if expected_type == 'css' and 'text/css' in content_type:
            extension = '.css'
        elif expected_type == 'css' and 'text/html' in content_type:
            extension = '.html'
        elif expected_type == 'js' and 'javascript' in content_type:
            extension = '.js'
        else:
            print(f'Skipped (Unknown Type): {file_url}')
            return

        filename = file_url.split('/')[-1].split('?')[0] or 'unnamed_file'
        if not filename.endswith(extension):
            filename += extension  

        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_response.text)

        print(f'Downloaded: {file_url} as {extension}')
    except Exception as e:
        print(f'Failed to download {file_url} - {e}')

# Read URLs from urls.txt
with open('urls.txt', 'r') as file:
    urls = [line.strip() for line in file if line.strip()]

output_dir = 'legitimate_pages'
os.makedirs(output_dir, exist_ok=True)

count = 8610
for url in urls:
    url = "https://" + url
    try:
        html_response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        html_response.raise_for_status()

        # Compute hash of the HTML content
        content_hash = compute_hash(html_response.text)

        # Create domain-based folder for saving files
        domain = urlparse(url).netloc.replace('www.', '')
        domain = str(count) + "_" + domain
        count += 1
        folder_path = os.path.join(output_dir, domain)

        # Skip folder creation and saving if it's a duplicate
        if is_duplicate(folder_path, content_hash):
            print(f'Duplicate found, skipping: {url}')
            continue

        # Only create folder if it's not a duplicate
        os.makedirs(folder_path, exist_ok=True)

        # Save HTML file
        html_path = os.path.join(folder_path, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_response.text)

        print(f'Successfully saved HTML: {url}')

        # Save the hash to avoid future duplicates
        save_hash(folder_path, content_hash)

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_response.text, 'html.parser')

        # Collect metadata
        title = soup.title.string if soup.title else 'No Title'
        num_links = len(soup.find_all('a'))
        num_images = len(soup.find_all('img'))
        contains_forms = bool(soup.find_all('form'))

        save_metadata(url, folder_path, title, num_links, num_images, contains_forms)

        # Download CSS and JS in parallel
        css_folder = os.path.join(folder_path, 'css')
        js_folder = os.path.join(folder_path, 'js')
        os.makedirs(css_folder, exist_ok=True)
        os.makedirs(js_folder, exist_ok=True)

        with ThreadPoolExecutor(max_workers=10) as executor:
            css_tasks = [executor.submit(download_file, urljoin(url, link.get('href')), css_folder, 'css')
                         for link in soup.find_all('link', rel='stylesheet')]
            js_tasks = [executor.submit(download_file, urljoin(url, script.get('src')), js_folder, 'js')
                        for script in soup.find_all('script', src=True)]

    except Exception as e:
        print(f'Failed to fetch {url} - {e}')
