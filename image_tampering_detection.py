import os
import json
import logging
from PIL import Image, ImageChops, ExifTags, ImageFilter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_ela(image_path, quality=90, output_path=None):
    """
    Apply Error Level Analysis (ELA) on the image to detect tampering.
    """
    logging.info(f"Applying ELA on: {image_path}")
    img = Image.open(image_path).convert('RGB')
    temp_path = 'temp_ela.jpg'
    img.save(temp_path, quality=quality)
    resaved = Image.open(temp_path)
    diff = ImageChops.difference(img, resaved)
    max_diff = max([ex[1] for ex in diff.getextrema()])
    ela_img = diff.point(lambda i: i * (255.0 / max_diff)).convert('L') if max_diff else Image.new('L', img.size)
    if output_path:
        ela_img.save(output_path)
        logging.info(f"ELA image saved to {output_path}")
    os.remove(temp_path)
    return np.array(ela_img)

def analyze_exif(image_path):
    """
    Analyze EXIF metadata of the image.
    """
    logging.info(f"Analyzing EXIF data for: {image_path}")
    exif_data, anomalies = {}, []
    try:
        img = Image.open(image_path)
        info = img._getexif() if hasattr(img, '_getexif') else None
        if info:
            for tag, val in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = val
            if 'Software' in exif_data and any(sw in exif_data['Software'].lower() for sw in ['photoshop', 'gimp']):
                anomalies.append(f"Editing software detected: {exif_data['Software']}")
            if 'DateTimeOriginal' not in exif_data and 'DateTime' in exif_data:
                anomalies.append("Missing original capture date")
            if 'Make' not in exif_data or 'Model' not in exif_data:
                anomalies.append("Missing camera make/model info")
        else:
            anomalies.append("No EXIF data found")
    except Exception as e:
        logging.error(f"EXIF analysis error: {e}")
        anomalies.append(f"EXIF error: {e}")
    return {"raw_exif": exif_data, "anomalies": anomalies, "has_exif_anomalies": bool(anomalies)}

def analyze_noise(image_path):
    """
    Analyze noise pattern to detect image inconsistencies.
    """
    logging.info(f"Analyzing noise pattern for: {image_path}")
    anomalies = []
    try:
        img = Image.open(image_path).convert('L')
        edges = np.abs(np.array(img.filter(ImageFilter.FIND_EDGES)))
        std_dev = np.std(edges)
        if std_dev < 5:
            anomalies.append(f"Low noise std deviation: {std_dev:.2f}")
    except Exception as e:
        logging.error(f"Noise analysis error: {e}")
        anomalies.append(f"Noise analysis error: {e}")
        std_dev = 0.0
    return {"noise_std_dev": float(std_dev), "anomalies": anomalies, "has_noise_anomalies": bool(anomalies)}

def detect_tampering(image_path, ela_quality=90, ela_threshold=0.5, ela_output=None):
    """
    Detect tampering by applying ELA, EXIF, and noise analysis.
    """
    logging.info(f"Starting tampering detection for: {image_path}")
    result = {"is_tampered": False, "tampering_details": []}

    try:
        ela_img = apply_ela(image_path, ela_quality, ela_output)
        bright_pct = (np.sum(ela_img > 230) / ela_img.size) * 100
        result["ela_analysis"] = {
            "mean_intensity": float(np.mean(ela_img)),
            "percentage_bright_pixels": bright_pct,
            "ela_anomaly_detected": bright_pct > ela_threshold
        }
        if bright_pct > ela_threshold:
            result["tampering_details"].append("ELA anomaly detected")
            result["is_tampered"] = True
    except Exception as e:
        logging.error(f"ELA error: {e}")
        result["ela_analysis"] = {"error": str(e)}

    exif = analyze_exif(image_path)
    result["exif_analysis"] = exif
    if exif["has_exif_anomalies"]:
        result["tampering_details"].extend(exif["anomalies"])
        result["is_tampered"] = True

    noise = analyze_noise(image_path)
    result["noise_analysis"] = noise
    if noise["has_noise_anomalies"]:
        result["tampering_details"].extend(noise["anomalies"])
        result["is_tampered"] = True

    logging.info(f"Tampering detection completed for: {image_path}. Tampered: {result['is_tampered']}")
    return result

if __name__ == '__main__':
    # Pass image path here
    input_path = "data\pan_card_fake.jpg"
    if not os.path.exists(input_path):
        logging.error(f"File does not exist: {input_path}")
    else:
        results = detect_tampering(input_path, ela_output='ela_output.jpg')
        print(json.dumps(results, indent=2))
