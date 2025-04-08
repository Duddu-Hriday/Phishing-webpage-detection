# 🧠 MLP-Based Phishing Webpage Detection

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)** model for detecting phishing webpages using handcrafted features extracted from HTML, URL, and associated resources.

The model is trained on a dataset of phishing and legitimate websites and achieves high accuracy using basic statistical and structural features.

## 📌 Highlights

- 🔍 Detects phishing websites based on extracted features
- 🏗️ Implements a custom Multi-Layer Perceptron (MLP) using PyTorch
- 📊 Supports training, validation, and performance evaluation
- 📁 Works with datasets collected using tools like Phish-Blitz
- 💾 Outputs performance metrics: accuracy, precision, recall, F1-score

## 🧬 Feature Extraction

The following features are used for classification:

- **URL-based features**:
  - URL length
  - Domain is IP
  - Number of subdomains
  - Suspicious Keywords
  - Symbol count
  - HTTPS
  - TLD in Domain

- **HTML based features**:
  - Number of Hyper links
  - Number of internal links
  - Number of external links
  - Number of script tags
  - Number of iframes
  - Has meta refresh
  - Hidden
  - Empty link
  - Login form
  - Internal External Resource
  - Redirect
  - Title Domain
  - Is link valid
  - Mulitple HTTS check
  - Form empty action
  - Is mail
  - Status bar customization
- **Javascript based features**:
   - Has eval
   - Has Document write
   - Has window open
   - Has set timeout
   - Has set interval
   - Has obfuscated code
   - Has event listeners
- **CSS based features**:
   - Number of CSS files
   - Number of external CSS
   - Number of important
   - Non standard colors
   - Unusual Font family
   - Use of display none
     
## ⚙️ Installation
---------------
1. Clone the repository
   ```
   git clone https://github.com/Duddu-Hriday/Phishing-webpage-detection.git
   cd '.\Phishing webpage detection\'
   ```

2. Install dependencies
   pip install -r requirements.txt

## 🚀 Leigitimate and Phishing dataset collection
  ```
cd '.\Data Collection\'
python extract_legitimate.py
python extract_phishing.py
  ```

## 🔧 Feature Extraction
  ```
cd '.\Data Collection\'
python legitimate_features.py
python phishing_features.py
  ```

## 🚀 Training the MLP
  ```
cd '.\Model and dataset\'
python mlp.py
  ```

## 📈 Evaluation
python evaluate.py --model saved_model.pth --data data/combined_features.csv

## 📊 Results
----------
| Metric     | Value     |
|------------|-----------|
| Accuracy   | 98.52%     |
| Precision  | 98.77%     |
| Recall     | 97.49%     |
| F1-Score   | 98.13%     |

## 📚 Dependencies
----------------
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- PyTorch
## 🤝 Contributing
----------------
Feel free to fork this repo, improve the model or feature set, and create pull requests. Suggestions and bug reports are welcome!

## 📜 License
-----------
This project is licensed under the MIT License.

## 🙋‍♂️ Author
------------
Duddu Hriday  
📧 student.hridayduddu@gmail.com 
🔗 https://linkedin.com/in/duddu-hriday

“Phishing is evolving — and so should our defenses. Detect it smarter with machine learning.”

