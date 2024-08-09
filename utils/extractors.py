import re
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import pytesseract  # For OCR from images

# Function to extract text from a given URL
async def extract_text_html(url: str) -> str:
    """
    Extracts and cleans text content from a given URL based on the detected language.
    """
    try:
        # Set headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Make an HTTP GET request
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text(separator=' ')

        # Detect language
        language = detect(text_content)

        # Clean the text content based on detected language
        clean_text_content = await clean_text(text_content, language)

        if len(clean_text_content) > 12500:
            # Handle very long texts
            pattern = r'\|([^|]+)\|'
            tokens = re.findall(pattern, clean_text_content)
            clean_text_content = " | ".join([word for word in tokens if len(word) < 50])

        return clean_text_content

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error processing content: {e}"

# Function to extract text from an image
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to extract URLs from a message
def extract_urls(message: str) -> list:
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(message)
