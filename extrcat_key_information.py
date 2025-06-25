import os
import logging
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

# ---------------- CONFIG ----------------
GOOGLE_API_KEY = "your_google_api_key_here"  # Replace with your Gemini API key
MODEL_NAME = "gemini-1.5-flash"  # Use 'gemini-1.5-pro' for higher reasoning
# ----------------------------------------

# ---------- Logging Setup ---------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------- Gemini Setup ------------------
def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        logging.info("Gemini model initialized successfully.")
        return model
    except Exception as e:
        logging.exception("Failed to configure Gemini.")
        raise

# -------- Load Image --------------------
def load_image_from_path_or_url(image_input):
    try:
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                logging.info(f"Downloading image from URL: {image_input}")
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                logging.info(f"Loading image from file: {image_input}")
                image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Unsupported image input type.")
        logging.info("Image loaded successfully.")
        return image.convert("RGB")
    except Exception as e:
        logging.exception("Error loading image.")
        raise

# -------- Call Gemini with Image --------
def extract_key_info_from_image(model, image):
    try:
        prompt = """
            You are an intelligent assistant. Extract all the key information from the given document image and return it as a clean Python dictionary.

            Only return the dictionary with appropriate key-value pairs such as:
            - DocumentType
            - Name
            - Date of Birth
            - Address
            - phone numbers
            - any other specific fields found

            Avoid any explanations or formatting other than the dictionary.
            """
        response = model.generate_content(
            contents=[prompt, image],
            stream=False
        )
        output_text = response.text.strip()

        # Attempt to safely evaluate as Python dict
        logging.info("Model responded. Attempting to parse result.")
        if output_text.startswith("{") and output_text.endswith("}"):
            try:
                import ast
                result = ast.literal_eval(output_text)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        # Fallback: return raw text
        logging.warning("Could not parse dictionary. Returning raw text.")
        return {"raw_output": output_text}
    except Exception as e:
        logging.exception("Failed to extract information from image.")
        return {"error": str(e)}

# -------- Main Logic --------------------
def main(image_input):
    try:
        model = setup_gemini(GOOGLE_API_KEY)
        image = load_image_from_path_or_url(image_input)
        info = extract_key_info_from_image(model, image)

        print("\nExtracted Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
    except Exception as e:
        logging.error("Pipeline failed.")

# --------- Example Execution ------------
if __name__ == "__main__":
    # Provide either a local path or image URL
    image_path_or_url = "data\driving_license1.jpeg"  # Replace with your image path or URL
    main(image_path_or_url)
