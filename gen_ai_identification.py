import os
import json
import google.generativeai as genai
from google.generativeai import types
from PIL import Image

class DocumentFraudChecker:
    """
    A class to perform fraudulent checks on document images using Gemini Flash.
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                           tools=[genai.Tool(code_execution={})])

    def _generate_prompt(self, task: str, document_type: str = "document"):
        """
        Generates specific prompts for each fraud detection task.
        """
        prompts = {
            "ai_generated": f"Analyze the provided {document_type} image. Is there any evidence that this document image was generated or heavily manipulated by an AI (e.g., deepfake, generative adversarial network)? Look for subtle artifacts, unusual patterns, or inconsistent details. Respond with only 'True' or 'False'.",
            "text_anomaly": f"Examine the text content on the provided {document_type} image. Are there any text anomalies present, such as inconsistent fonts, varying text sizes, misalignments, unusual spacing, grammatical errors, or suspicious phrasing that would indicate tampering or fraud? Provide specific examples if found. Respond with a JSON object: {{'has_anomaly': boolean, 'details': '...'}}.",
            "mathematical_validation": f"Perform mathematical and logical validation on the numerical and textual data present in the provided {document_type} image. For example, if it's an Aadhar card, check for valid Aadhar number format and checksum. If it's a financial document, identify any sums, dates, or account numbers and assess their consistency. If a mortgage document, check if principal, interest, and term align with typical calculations (you might need to provide specific values for this, or the model will make assumptions). Respond with a JSON object: {{'is_mathematically_valid': boolean, 'details': '...'}}. If the document type requires specific mathematical rules, assume standard validation for common documents like Aadhar or PAN, or state if the model cannot perform a specific check.",
            "watermark_detection": f"Inspect the provided {document_type} image for any visible or hidden watermarks. Describe the characteristics of any detected watermarks (e.g., faint text, logo, pattern). If no watermark is detected, state that. Respond with a JSON object: {{'watermark_detected': boolean, 'description': '...'}}."
        }
        return prompts.get(task)

    def analyze_document(self, image_path: str, document_type: str = "document") -> dict:
        """
        Performs all fraud checks on a given document image.

        Args:
            image_path (str): The path to the document image.
            document_type (str): The type of document (e.g., "Aadhar card", "PAN card", "mortgage document").

        Returns:
            dict: A dictionary containing boolean results for each check.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return {"error": f"Image file not found at: {image_path}"}
        except Exception as e:
            return {"error": f"Error loading image: {e}"}

        results = {}

        # 1. AI Generated Check
        try:
            prompt_ai_gen = self._generate_prompt("ai_generated", document_type)
            response_ai_gen = self.model.generate_content(
                contents=[prompt_ai_gen, img]
            )
            # Attempt to parse boolean directly, handle potential extra text
            ai_generated_str = response_ai_gen.text.strip().lower()
            results['is_ai_generated'] = 'true' in ai_generated_str and 'false' not in ai_generated_str

        except Exception as e:
            results['is_ai_generated'] = False
            print(f"Error during AI generated check: {e}")

        # 2. Text Anomaly Check
        try:
            prompt_text_anomaly = self._generate_prompt("text_anomaly", document_type)
            response_text_anomaly = self.model.generate_content(
                contents=[prompt_text_anomaly, img],
                generation_config=types.GenerationConfig(response_mime_type="application/json")
            )
            text_anomaly_data = json.loads(response_text_anomaly.text)
            results['has_text_anomaly'] = text_anomaly_data.get('has_anomaly', False)
            results['text_anomaly_details'] = text_anomaly_data.get('details', 'N/A')
        except Exception as e:
            results['has_text_anomaly'] = False
            results['text_anomaly_details'] = f"Error during text anomaly check: {e}"
            print(f"Error during text anomaly check: {e}")

        # 3. Mathematical Validation
        try:
            prompt_math_validation = self._generate_prompt("mathematical_validation", document_type)
            response_math_validation = self.model.generate_content(
                contents=[prompt_math_validation, img],
                generation_config=types.GenerationConfig(response_mime_type="application/json")
            )
            math_validation_data = json.loads(response_math_validation.text)
            results['is_mathematically_valid'] = math_validation_data.get('is_mathematically_valid', False)
            results['mathematical_validation_details'] = math_validation_data.get('details', 'N/A')
        except Exception as e:
            results['is_mathematically_valid'] = False
            results['mathematical_validation_details'] = f"Error during mathematical validation: {e}"
            print(f"Error during mathematical validation: {e}")

        # 4. Watermark Detection
        try:
            prompt_watermark = self._generate_prompt("watermark_detection", document_type)
            response_watermark = self.model.generate_content(
                contents=[prompt_watermark, img],
                generation_config=types.GenerationConfig(response_mime_type="application/json")
            )
            watermark_data = json.loads(response_watermark.text)
            results['watermark_detected'] = watermark_data.get('watermark_detected', False)
            results['watermark_description'] = watermark_data.get('description', 'N/A')
        except Exception as e:
            results['watermark_detected'] = False
            results['watermark_description'] = f"Error during watermark detection: {e}"
            print(f"Error during watermark detection: {e}")

        return results

# --- Example Usage ---
if __name__ == "__main__":
    # Set your Google Gemini API Key here or as an environment variable
    # os.environ.get("GOOGLE_API_KEY") is recommended for production
    API_KEY = os.getenv("GOOGLE_API_KEY") 
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set your API key.")

    checker = DocumentFraudChecker(api_key=API_KEY)

    # Replace with the actual path to your document image
    # Example: "path/to/your/aadhar_card.png" or "path/to/your/mortgage_document.jpg"
    document_image_path = "data\adhar_1.jpeg" 
    
    # Specify the type of document for more tailored analysis
    # Examples: "Aadhar card", "PAN card", "Driving License", "Mortgage Document", "Bank Statement"
    doc_type = "Aadhar card" 

    print(f"Analyzing document: {document_image_path} (Type: {doc_type})")
    fraud_check_results = checker.analyze_document(document_image_path, doc_type)

    print("\n--- Fraud Check Results ---")
    print(json.dumps(fraud_check_results, indent=4))

    # You can then use these boolean values for further logic in your banking system
    if fraud_check_results.get('is_ai_generated'):
        print("\nALERT: Document appears to be AI-generated!")
    if fraud_check_results.get('has_text_anomaly'):
        print(f"WARNING: Text anomalies detected: {fraud_check_results.get('text_anomaly_details')}")
    if not fraud_check_results.get('is_mathematically_valid'):
        print(f"WARNING: Mathematical validation failed: {fraud_check_results.get('mathematical_validation_details')}")
    if fraud_check_results.get('watermark_detected'):
        print(f"INFO: Watermark detected: {fraud_check_results.get('watermark_description')}")
    else:
        print("INFO: No watermark explicitly detected.")