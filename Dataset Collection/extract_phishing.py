import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, urljoin
from io import StringIO
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Function to compute the hash of HTML content
def compute_hash(html_content):
    return hashlib.md5(html_content.encode('utf-8')).hexdigest()

# URL and Headers
url = 'https://phishstats.info/phish_score.csv'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
response.raise_for_status()

# Parse CSV from the response
data = pd.read_csv(StringIO(response.text), comment='#', header=None, names=['Date', 'Score', 'URL', 'IP'])
phishing_urls = data[data['Score'] > 4]['URL'].head(20000).tolist()

output_dir = 'phishing_pages'
os.makedirs(output_dir, exist_ok=True)

# Function to download resources (CSS/JS) with dynamic extension
def download_file(file_url, folder_path, expected_type):
    try:
        file_response = requests.get(file_url, headers=headers, timeout=10)
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

# Check if a page is a duplicate based on its hash
def is_duplicate(folder_path, content_hash):
    hash_file = os.path.join(folder_path, 'hashes.txt')

    if os.path.exists(hash_file):
        with open(hash_file, 'r', encoding='utf-8') as file:
            existing_hashes = file.readlines()
            if content_hash + '\n' in existing_hashes:
                return True
    return False

# Function to save the page's hash to avoid duplicates in future
def save_hash(folder_path, content_hash):
    hash_file = os.path.join(folder_path, 'hashes.txt')
    with open(hash_file, 'a', encoding='utf-8') as file:
        file.write(content_hash + '\n')

# Check if HTML response is valid
def is_valid_html(content):
    if not content:
        return False

    error_messages = ["404 Not Found", "Country Not Found", "Invalid URL", "Error"]
    if any(error_message in content for error_message in error_messages):
        return False

    return True

# Function to fetch and save CSS/JS resources in parallel
def fetch_resources(soup, url, css_folder, js_folder):
    with ThreadPoolExecutor(max_workers=10) as executor:
        for link in soup.find_all('link', rel='stylesheet'):
            css_url = urljoin(url, link.get('href'))
            executor.submit(download_file, css_url, css_folder, 'css')

        for script in soup.find_all('script', src=True):
            js_url = urljoin(url, script.get('src'))
            executor.submit(download_file, js_url, js_folder, 'js')

existing_urls = set()
if os.path.exists('phishing_urls.txt'):
    with open('phishing_urls.txt', 'r', encoding='utf-8') as file:
        existing_urls = set(line.strip() for line in file)

count = 5663
# Main loop to process phishing URLs
for url in phishing_urls:
    
    if url in existing_urls:  # Skip if URL already present
        print(f'URL already processed, skipping: {url}')
        continue
    try:
        html_response = requests.get(url, headers=headers, timeout=10)
        html_response.raise_for_status()

        if not is_valid_html(html_response.text):
            print(f"Invalid or empty HTML response, skipping: {url}")
            continue

        domain = urlparse(url).netloc.replace('www.', '')
        domain = str(count) + "_" + domain
        count += 1
        folder_path = os.path.join(output_dir, domain)

        os.makedirs(folder_path, exist_ok=True)

        content_hash = compute_hash(html_response.text)

        if is_duplicate(folder_path, content_hash):
            print(f'Duplicate found, skipping: {url}')
            continue

        # Save HTML file
        html_path = os.path.join(folder_path, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_response.text)

        print(f'Successfully saved HTML: {url}')
        
        with open('phishing_urls.txt', 'a', encoding='utf-8') as file:
            file.write(url + '\n')

        save_hash(folder_path, content_hash)

        soup = BeautifulSoup(html_response.text, 'html.parser')

        title = soup.title.string if soup.title else 'No Title'
        num_links = len(soup.find_all('a'))
        num_images = len(soup.find_all('img'))
        contains_forms = bool(soup.find_all('form'))

        save_metadata(url, folder_path, title, num_links, num_images, contains_forms)

        css_folder = os.path.join(folder_path, 'css')
        js_folder = os.path.join(folder_path, 'js')
        os.makedirs(css_folder, exist_ok=True)
        os.makedirs(js_folder, exist_ok=True)

        # Parallel downloading of CSS and JS
        fetch_resources(soup, url, css_folder, js_folder)

    except Exception as e:
        print(f'Failed to fetch {url} - {e}')
