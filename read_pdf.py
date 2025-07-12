#!/usr/bin/env python3
import re

def extract_text_from_pdf(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    
    # Find text between BT (Begin Text) and ET (End Text) markers
    text_pattern = rb'BT\s*(.*?)\s*ET'
    matches = re.findall(text_pattern, content, re.DOTALL)
    
    extracted_text = []
    for match in matches:
        # Decode text, handling PDF escape sequences
        try:
            text = match.decode('utf-8', errors='ignore')
            # Clean up PDF commands
            text = re.sub(r'\s*\d+\s+\d+\s+Td\s*', ' ', text)
            text = re.sub(r'\s*Tf\s*', '', text)
            text = re.sub(r'[\(\)]', '', text)
            text = text.strip()
            if text:
                extracted_text.append(text)
        except:
            pass
    
    return '\n'.join(extracted_text)

if __name__ == "__main__":
    text = extract_text_from_pdf('patterns.pdf')
    print(text[:2000] if text else "No text found")