"""
Setup script for NLP components
"""

import nltk
import subprocess
import sys

def setup_nlp():
    """Download required NLP models and data"""
    
    print("📥 Downloading NLP components...")
    
    # Download TextBlob corpora
    print("1. Setting up TextBlob...")
    try:
        subprocess.run([sys.executable, "-m", "textblob.download_corpora"], check=True)
        print("✅ TextBlob data downloaded")
    except:
        print("⚠️ TextBlob download failed - run 'python -m textblob.download_corpora' manually")
    
    # Download NLTK data
    print("\n2. Setting up NLTK...")
    nltk_downloads = ['punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger']
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
            print(f"✅ Downloaded {item}")
        except:
            print(f"⚠️ Failed to download {item}")
    
    print("\n🎉 NLP setup complete!")
    print("\n💡 Note: SpaCy models are lightweight and loaded on-demand.")
    
if __name__ == "__main__":
    setup_nlp()