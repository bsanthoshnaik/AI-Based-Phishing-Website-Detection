import re
import tldextract
import pandas as pd

def extract_features(url: str):
    """Extract 30 phishing detection features matching training dataset (offline-safe)."""
    features = {}

    
    # 1️⃣ URL-based lexical features
    
    # Having IP address
    features['having_IP_Address'] = -1 if re.match(r'(\d{1,3}\.){3}\d{1,3}', url) else 1

    # URL length
    features['URL_Length'] = 1 if len(url) < 54 else 0 if len(url) <= 75 else -1

    # Shortening service
    shortening_services = r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly"
    features['Shortining_Service'] = -1 if re.search(shortening_services, url) else 1

    # Having '@' symbol
    features['having_At_Symbol'] = -1 if '@' in url else 1

    # Double slash redirecting
    features['double_slash_redirecting'] = -1 if url.count('//') > 1 else 1

    # Prefix-Suffix in domain
    domain = tldextract.extract(url).domain
    features['Prefix_Suffix'] = -1 if '-' in domain else 1

    # Having Subdomain
    subdomains = tldextract.extract(url).subdomain.split('.')
    features['having_Sub_Domain'] = -1 if len(subdomains) > 2 else (0 if len(subdomains) == 1 else 1)

    # SSLfinal_State (https check)
    features['SSLfinal_State'] = 1 if url.startswith("https") else -1

    
    # 2️⃣ Placeholder (offline) domain/content
    
    features['Domain_registeration_length'] = 0  # WHOIS required
    features['Favicon'] = 0
    features['port'] = 0
    features['HTTPS_token'] = 0
    features['Request_URL'] = -1 if re.search(r'(login|secure|account|update)', url) else 0
    features['URL_of_Anchor'] = 0
    features['Links_in_tags'] = 0
    features['SFH'] = 0
    features['Submitting_to_email'] = -1 if 'mail()' in url or 'mailto:' in url else 0
    features['Abnormal_URL'] = -1 if re.search(r'//.*//', url) else 0
    features['Redirect'] = 0
    features['on_mouseover'] = 0
    features['RightClick'] = 0
    features['popUpWidnow'] = 0
    features['Iframe'] = 0
    features['age_of_domain'] = 0
    features['DNSRecord'] = 0
    features['web_traffic'] = 0
    features['Page_Rank'] = 0
    features['Google_Index'] = 0
    features['Links_pointing_to_page'] = 0
    features['Statistical_report'] = 0

    
    # 3️⃣ Ensure correct feature order
    
    expected_cols = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain',
        'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port',
        'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
        'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover',
        'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord',
        'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]

    df = pd.DataFrame([[features[col] for col in expected_cols]], columns=expected_cols)
    return df
