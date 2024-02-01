import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

nltk.download('punkt')

def summarize_website(url, num_sentences=3):
    # Step 1: Text Extraction
    response = requests.get(url)
    html_content = response.text

    # Step 2: Preprocessing
    soup = BeautifulSoup(html_content, 'html.parser')
    clean_text = soup.get_text()

    # Step 3: Tokenization
    tokens = word_tokenize(clean_text)

    # Step 4: NLP Analysis (Part-of-speech tagging and Named Entity Recognition)
    pos_tags = nltk.pos_tag(tokens)

    # Step 6: Summarization Algorithms (LSA)
    parser = PlaintextParser.from_string(clean_text, Tokenizer('english'))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)

    # Step 8: Output
    result = '\n'.join(str(sentence) for sentence in summary)
    return result

# Example usage:
website_url = 'https://www.boat-lifestyle.com/'
summary_result = summarize_website(website_url)
print(summary_result)
