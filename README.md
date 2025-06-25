# Fraud Document Detection System

## Overview

This project aims to detect fraudulent documents using an ensemble of techniques that include image analysis, metadata inspection, text-based similarity, and AI-driven classification. Each technique is implemented as a modular Python script, allowing for flexibility and integration into larger document verification systems.

The system is applicable in domains such as KYC, financial onboarding, education, HR, and government verification workflows.

---

## Fraud Detection Modules

Below is a summary of each fraud detection method implemented in the system along with its corresponding Python script.

### 1. Extract Key Information and Store in Database  
- **Script**: `extract_key_information.py`  
- **Description**: Extracts important fields like name, date of birth, ID numbers, and addresses using OCR and stores the data in a structured format for validation.  
- **Application**: Enables downstream analytics, cross-document matching, and storage in databases for historical or real-time checks.

---

### 2. Anomaly Detection (Using Generative AI Features)  
- **Script**: `gen_ai_identification.py`  
- **Description**: Identifies AI-generated elements, such as synthetic text, faces, or formatting inconsistencies that may indicate a manipulated document.  
- **Application**: Useful for detecting fake government IDs, AI-generated resumes, or altered certificates.

---

### 3. Metadata Analysis  
- **Script**: `image_tampering_detection.py`  
- **Description**: Extracts and analyzes EXIF metadata like software used, device make/model, and timestamps. Highlights anomalies like missing or conflicting metadata.  
- **Application**: Detects documents altered using image editing tools like Photoshop or GIMP.

---

### 4. Tampering Detection (Error Level Analysis - ELA)  
- **Script**: `image_tampering_detection.py`  
- **Description**: Performs Error Level Analysis to identify regions with inconsistent compression, indicating possible edits or pasted sections.  
- **Application**: Useful for identifying manipulated regions in scanned certificates, ID cards, or documents with added signatures or photos.

---

### 5. Watermark and Overlay Detection  
- **Script**: `gen_ai_identification.py`  
- **Description**: Detects missing or forged watermarks and overlays commonly found in genuine documents.  
- **Application**: Applicable for verifying authenticity in invoices, university transcripts, and government-issued papers.

---

### 6. AI-Based Image Classification  
- **Script**: `AI_Image_Classification.py`  
- **Description**: Classifies images as either real or AI-generated using a deep learning model.  
- **Application**: Screens for completely fabricated or manipulated documents created via generative tools.

---

### 7. Textual Anomaly Detection  
- **Script**: `gen_ai_identification.py`  
- **Description**: Detects linguistic inconsistencies, unusual formatting, or auto-generated content in documents.  
- **Application**: Identifies machine-generated content in fake certificates, fake resumes, or IDs with fabricated addresses.

---

### 8. Image Similarity Comparison  
- **Script**: `image_similarity.py`  
- **Description**: Measures similarity between two document images using visual feature embeddings.  
- **Application**: Flags duplicate or reused document templates with minor alterations.

---

### 9. Textual Similarity Comparison  
- **Script**: `text_similarity.py`  
- **Description**: Uses OCR and sentence embeddings to compare the textual content of two document images.  
- **Application**: Detects copied or slightly modified versions of documents that reuse core text structure.

---

## Next Steps and Future Improvements

Here are some proposed improvements to enhance the fraud detection system:

### 1. Train a Custom CNN Model for Document Classification  
- Fine-tune a convolutional neural network on real-world datasets of authentic and fraudulent documents to improve detection precision.

### 2. Cross-Document Information Validation  
- Compare extracted key fields (name, DOB, ID number) across multiple submitted documents (e.g., Aadhar vs PAN) to catch inconsistencies.

### 3. Localized Text & Language Support  
- Expand OCR capabilities to support Indian regional languages (e.g., Hindi, Marathi, Tamil) for broader coverage.


---

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Neelesh1121/document_fraud_classification.git
    cd document_fraud_classification
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run individual modules:
    ```bash
    python extract_key_information.py --image path/to/document.jpg
    python gen_ai_identification.py --image path/to/document.jpg
    python image_tampering_detection.py --image path/to/document.jpg
    ```

4. Combine results using your own orchestration script or pipeline.

---

## License

This project is intended for research and internal evaluation purposes. For commercial use, please contact the project maintainers.

---

## Contact

For issues, improvements, or collaboration, please reach out via GitHub Issues or email: [neelesh932@gmail.com]


