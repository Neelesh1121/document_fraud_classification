import easyocr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import os
import argparse
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LANG_LIST = ['en'] 
MODEL_NAME = 'all-MiniLM-L6-v2' 
SIMILARITY_THRESHOLD = 0.85

def extract_text_from_image(image_path, reader):
    """
    Extract text from an image using EasyOCR.
    """
    try:
        results = reader.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])
        return extracted_text.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    except Exception as e:
        raise ValueError(f"Error extracting text from {image_path}: {e}")

def get_text_embedding(text, model):
    """
    Generate embedding vector for a given text.
    """
    if not text:
        return np.array([])
    return model.encode(text)

def compare_textual_content(reference_image_path, validating_image_path):
    """
    Compare text from two images and determine similarity.
    """
    results = {
        "reference_text": "",
        "validating_text": "",
        "similarity_score": 0.0,
        "is_textually_similar": False,
        "threshold_used": SIMILARITY_THRESHOLD,
        "error": None
    }

    # Initialize OCR and model
    logging.info(f"Initializing OCR with languages: {LANG_LIST}")
    reader = easyocr.Reader(LANG_LIST)
    model = SentenceTransformer(MODEL_NAME)

    try:
        logging.info(f"Extracting text from reference image: {reference_image_path}")
        ref_text = extract_text_from_image(reference_image_path, reader)
        results["reference_text"] = ref_text

        logging.info(f"Extracting text from validating image: {validating_image_path}")
        val_text = extract_text_from_image(validating_image_path, reader)
        results["validating_text"] = val_text

        if not ref_text or not val_text:
            results["error"] = "Insufficient text extracted from one or both images."
            return results

        ref_embedding = get_text_embedding(ref_text, model)
        val_embedding = get_text_embedding(val_text, model)

        score = cosine_similarity(ref_embedding.reshape(1, -1), val_embedding.reshape(1, -1))[0][0]
        results["similarity_score"] = float(score)
        results["is_textually_similar"] = score >= SIMILARITY_THRESHOLD

    except Exception as e:
        results["error"] = str(e)
        logging.error(f"An error occurred: {e}")

    return resultss

    

if __name__ == "__main__":
    
    
    ref_path = 'data\adhar_1.jpeg'
    inference_path = 'data\pan_card_fake.jpg'
    results = compare_textual_content(
        reference_image_path=ref_path,
        validating_image_path=inference_path,
    )

    logging.info("\nFinal Results:")
    for k, v in results.items():
        logging.info(f"{k}: {v}")
