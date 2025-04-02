import requests
from bs4 import BeautifulSoup
import os
import tempfile  # Added for temporary directory
from urllib.parse import urlparse, urljoin
import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
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

# URL-Based Features
def extract_url_features(url):
    parsed_url = urlparse(url)
    return {
        'url_length': len(url),
        'domain_is_IP': any(char.isdigit() for char in parsed_url.netloc),
        'num_subdomains': parsed_url.netloc.count('.'),
        # 'has_at_symbol': '@' in url,
        'suspicious_keywords': any(keyword in url.lower() for keyword in ['secure', 'account', 'login']),
        # 'tld': parsed_url.netloc.split('.')[-1],
        'symbol_count': sum(char in ['@', '-', '_'] for char in url),
        'https': url.startswith('https'),
        'tld_in_domain': any(part in parsed_url.netloc for part in ['.com', '.net', '.org']),
        #'tld_in_path': any(part in parsed_url.path for part in ['.com', '.net', '.org']),
        #'https_in_domain': 'https' in parsed_url.netloc,
        #'abnormal_url': parsed_url.netloc != parsed_url.hostname,
    }

# HTML-Based Features
def extract_html_features(index_html_path):
    try:
        with open(index_html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # Length of HTML tags
        total_html_tag_length = sum(len(str(tag)) for tag in soup.find_all(['style', 'link', 'form', 'script']))

        return {
            'num_hyperlinks': len(soup.find_all('a')),
            'num_internal_links': len([a for a in soup.find_all('a') if a.get('href', '').startswith('/')]),
            'num_external_links': len([a for a in soup.find_all('a') if 'http' in a.get('href', '')]),
            'num_script_tags': len(soup.find_all('script')),
            'num_iframes': len(soup.find_all('iframe')),
            'has_meta_refresh': bool(soup.find('meta', attrs={'http-equiv': 'refresh'})),
            #'has_empty_form_action': any(form.get('action') in ['', 'about:blank'] for form in soup.find_all('form')),
            'hidden': any(tag.get('type') == 'hidden' for tag in soup.find_all(['div', 'input', 'button'])),
            'empty_link': len([a for a in soup.find_all('a') if a.get('href') in ['#', 'javascript::void(0)']]),
            'login_form': any(input_tag.get('name') in ['password', 'pass', 'login', 'signin'] for input_tag in soup.find_all('input')),
            'internal_external_resource': len(soup.find_all(['link', 'img', 'script', 'noscript'])),
            'redirect': 'redirect' in soup.get_text().lower(),
            #'alarm_window': any('alert' in str(tag) or 'window.open' in str(tag) for tag in soup.find_all('script')),
            'title_domain': any(tag.text.strip() in index_html_path for tag in soup.find_all('title')),
            #'brand_freq_domain': sum(brand in index_html_path for brand in ['google', 'facebook', 'paypal', 'amazon']),
            'is_link_valid': any(a.get('href', '').startswith('http') for a in soup.find_all('a')),
            'multiple_https_check': sum('https://' in a.get('href', '') for a in soup.find_all('a')) > 1,
            'form_empty_action': any(form.get('action') in ['', 'about:blank'] for form in soup.find_all('form')),
            'is_mail': any('mailto:' in a.get('href', '') for a in soup.find_all('a')),
            'status_bar_customization': any('onmouseover' in tag.attrs or 'window.status' in str(tag) for tag in soup.find_all())
        }
    except OSError:
        print(f"Skipping {index_html_path} due to OSError")
        return {}  # Return empty dictionary instead of None

# JavaScript-Based Features
def extract_js_features(js_folder_path):
    js_features = {
        #'num_js_files': 0,
        #'num_external_js': 0,
        'has_eval': 0,
        'has_document_write': 0,
        'has_window_open': 0,
        'has_setTimeout': 0,
        'has_setInterval': 0,
        'has_obfuscated_code': 0,
        'has_event_listeners': 0
    }

    try:
        for js_file in os.listdir(js_folder_path):
            js_file_path = os.path.join(js_folder_path, js_file)
            if not js_file.endswith('.js'):
                continue

            # js_features['num_js_files'] += 1

            with open(js_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                js_content = f.read()

            # Check for external JS files (if any are referenced in the current JS)
            # if 'http' in js_content or '.js' in js_content:
            #     js_features['num_external_js'] += 1

            # Check for suspicious functions
            js_features['has_eval'] += len(re.findall(r'\beval\(', js_content))
            js_features['has_document_write'] += len(re.findall(r'\bdocument\.write\(', js_content))
            js_features['has_window_open'] += len(re.findall(r'\bwindow\.open\(', js_content))
            js_features['has_setTimeout'] += len(re.findall(r'\bsetTimeout\(', js_content))
            js_features['has_setInterval'] += len(re.findall(r'\bsetInterval\(', js_content))

            # Check for obfuscated code by looking for base64 or hex encoding patterns
            if re.search(r'([A-Za-z0-9+/=]{10,})', js_content):
                js_features['has_obfuscated_code'] += 1

            # Check for event listeners
            js_features['has_event_listeners'] += len(re.findall(r'addEventListener\(', js_content))

    except OSError:
        print(f"Skipping {js_folder_path} due to OSError")

    return js_features

# CSS-Based Features
def extract_css_features(css_folder_path):
    css_features = {
        'num_css_files': 0,
        'num_external_css': 0,
        #'num_inline_css': 0,
        'num_important': 0,
        #'num_hidden_elements': 0,
        'non_standard_colors': 0,
        'unusual_font_family': 0,
        'use_of_display_none': 0
    }

    try:
        for css_file in os.listdir(css_folder_path):
            css_file_path = os.path.join(css_folder_path, css_file)
            if not css_file.endswith('.css'):
                continue

            css_features['num_css_files'] += 1

            with open(css_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                css_content = f.read()

            # Check for external CSS files (if any are linked in the current CSS)
            if 'http' in css_content or '.css' in css_content:
                css_features['num_external_css'] += 1

            # Check for inline CSS (if any styles are inside the HTML)
            if '<style>' in css_content:
                css_features['num_inline_css'] += 1

            # Check for !important usage
            css_features['num_important'] += len(re.findall(r'!important', css_content))

            # Check for hidden elements
            # css_features['num_hidden_elements'] += len(re.findall(r'visibility\s*:\s*hidden', css_content))
            # css_features['num_hidden_elements'] += len(re.findall(r'display\s*:\s*none', css_content))

            # Check for non-standard colors (like bright, flashing colors)
            css_features['non_standard_colors'] += len(re.findall(r'#[A-Fa-f0-9]{6}', css_content))

            # Check for unusual font family (often used for phishing)
            css_features['unusual_font_family'] += len(re.findall(r'font-family\s*:\s*(?!serif|sans-serif)', css_content))

            # Check for use of display: none or visibility: hidden
            css_features['use_of_display_none'] += len(re.findall(r'display\s*:\s*none', css_content))

    except OSError:
        print(f"Skipping {css_folder_path} due to OSError")

    return css_features

# Feature Extraction from Phishing Pages
def extract_features(phishing_pages_dir):
    features_list = []

    for folder in os.listdir(phishing_pages_dir):
        folder_path = os.path.join(phishing_pages_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        metadata_path = os.path.join(folder_path, 'metadata.json')
        index_html_path = os.path.join(folder_path, 'index.html')
        js_folder_path = os.path.join(folder_path, 'js')
        css_folder_path = os.path.join(folder_path, 'css')

        if not os.path.exists(metadata_path) or not os.path.exists(index_html_path) or not os.path.exists(js_folder_path) or not os.path.exists(css_folder_path):
            continue  # Skip folders missing necessary files

        with open(metadata_path, 'r') as meta_file:
            metadata = json.load(meta_file)
            url = metadata.get("URL", "")

        features = extract_url_features(url)
        features.update(extract_html_features(index_html_path))
        features.update(extract_js_features(js_folder_path))  # Add JavaScript features
        features.update(extract_css_features(css_folder_path))  # Add CSS features
        # features['folder_name'] = folder
        features_list.append(features)
        print("Done folder:" + folder)

    return pd.DataFrame(features_list)



# Temporary directory for output
output_dir = tempfile.mkdtemp()  # Changed output to a temporary location
print(f"Temporary Directory: {output_dir}")

count = 8610
url = input("Enter the URL: ").strip()  # Accept URL from user
url = "https://" + url

try:
    html_response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    html_response.raise_for_status()

    domain = urlparse(url).netloc.replace('www.', '')
    domain = str(count) + "_" + domain
    count += 1
    folder_path = os.path.join(output_dir, domain)

    os.makedirs(folder_path, exist_ok=True)
    # Save HTML file
    html_path = os.path.join(folder_path, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as file:
        file.write(html_response.text)
    print(f'Successfully saved HTML: {url}')
    # Save the hash to avoid future duplicates
    # save_hash(folder_path, content_hash)
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


    features_df = extract_features(output_dir)
    features_df = features_df.replace({True: 1, False: 0})

    # features_df.to_csv('temp_features.csv', index=False)
    # data = pd.read_csv('temp_features.csv')
    # X = data.values
    features = features_df.select_dtypes(include=[np.number]).values

    # print("Feature extraction complete. Data saved to 'temp_features.csv'.")
    data = pd.read_csv('dataset.csv')
    X = data.drop('label', axis=1).values

    # Load and fit the same scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Load Model
    model = MLP(X.shape[1])
    model.load_state_dict(torch.load('best_mlp_model.pth'))
    model.eval()

    # New Input Data
    new_input = np.array(features)

    # Normalize and convert to tensor
    new_input = scaler.transform(new_input)
    print(new_input)
    new_input_tensor = torch.tensor(new_input, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        output = model(new_input_tensor).item()
        predicted_label = 1 if output > 0.5 else 0

    print(f"🔎 Predicted Label: {predicted_label}")

except Exception as e:
    print(f'Failed to fetch {url} - {e}')
