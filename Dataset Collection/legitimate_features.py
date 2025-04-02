import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

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

            js_features['num_js_files'] += 1

            with open(js_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                js_content = f.read()

            # Check for external JS files (if any are referenced in the current JS)
            if 'http' in js_content or '.js' in js_content:
                js_features['num_external_js'] += 1

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
            css_features['num_hidden_elements'] += len(re.findall(r'visibility\s*:\s*hidden', css_content))
            css_features['num_hidden_elements'] += len(re.findall(r'display\s*:\s*none', css_content))

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

# Directory containing phishing webpage folders
phishing_pages_dir = 'legitimate_pages'

# Extract features and save to CSV
features_df = extract_features(phishing_pages_dir)
features_df.to_csv('legitimate_features.csv', index=False)
print("Feature extraction complete. Data saved to 'phishing_features_with_css.csv'.")
