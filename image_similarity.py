import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the specified path and resizes it to 224x224 pixels.
    This is a standard input size for many pre-trained CNNs like VGG16.

    Args:
        image_path (str): The file path to the input image.

    Returns:
        PIL.Image.Image: The resized PIL Image object.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: For other errors during image loading or resizing.
    """
    logging.info(f"Attempting to load and resize image: {image_path}")
    try:
        input_image = Image.open(image_path)
        # Ensure the image is in RGB format (3 channels)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        resized_image = input_image.resize((224, 224))
        logging.info(f"Successfully loaded and resized image: {image_path}")
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at path: {image_path}", exc_info=True)
        raise FileNotFoundError(f"Image not found: {image_path}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}", exc_info=True)
        raise Exception(f"Failed to load or resize image {image_path}: {e}")


def get_image_embeddings(object_image: Image.Image, model: VGG16) -> np.ndarray:
    """
    Converts a PIL Image object into a 4D NumPy array suitable for model input
    and computes its feature embeddings using the provided VGG16 model.

    Args:
        object_image (PIL.Image.Image): The PIL Image object to process.
        model (tensorflow.keras.applications.vgg16.VGG16): The pre-loaded VGG16 model
                                                            used for feature extraction.

    Returns:
        numpy.ndarray: The feature embedding of the given image, a 2D array.

    Raises:
        Exception: For errors during image array conversion or model prediction.
    """
    logging.info("Converting image to array and getting embeddings.")
    try:
        # Convert PIL image to NumPy array and expand dimensions for model input (batch size)
        # image.img_to_array converts to (height, width, channels)
        # np.expand_dims adds a batch dimension (1, height, width, channels)
        image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
        
        # Predict embeddings using the VGG16 model
        image_embedding = model.predict(image_array)
        logging.info(f"Successfully obtained image embedding with shape: {image_embedding.shape}")
        return image_embedding
    except Exception as e:
        logging.error(f"Error getting image embeddings: {e}", exc_info=True)
        raise Exception(f"Failed to get image embeddings: {e}")


def calculate_cosine_similarity_score(first_image_embedding: np.ndarray, second_image_embedding: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two image embeddings.

    Args:
        first_image_embedding (numpy.ndarray): The embedding of the first image.
        second_image_embedding (numpy.ndarray): The embedding of the second image.

    Returns:
        float: The cosine similarity score, ranging from -1 (completely dissimilar) to 1 (identical).
    """
    logging.info("Calculating cosine similarity score.")
    try:
        # cosine_similarity returns a 2D array [[score]], reshape to get a scalar
        similarity_score = cosine_similarity(first_image_embedding, second_image_embedding).reshape(1,)[0]
        logging.info(f"Cosine similarity score computed: {similarity_score:.4f}")
        return similarity_score
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        raise Exception(f"Failed to calculate similarity score: {e}")


def perform_image_similarity_check(first_image_path: str, second_image_path: str) -> float:
    """
    Takes two image paths, loads them, computes their embeddings using a VGG16 model,
    and then calculates their cosine similarity score.

    Args:
        first_image_path (str): The file path to the first image.
        second_image_path (str): The file path to the second image.

    Returns:
        float: The similarity score between the two images.

    Raises:
        Exception: If any step in the process (loading, embedding, similarity calculation) fails.
    """
    logging.info(f"Starting image similarity check between '{first_image_path}' and '{second_image_path}'")
    
    try:
        # Initialize VGG16 model
        # weights='imagenet': Use pre-trained weights from ImageNet
        # include_top=False: Exclude the final classification layers, we only need features
        # pooling='max': Apply max pooling to the last convolutional block's output
        # input_shape=(224, 224, 3): Define the expected input image shape
        vgg16_model = VGG16(weights='imagenet', include_top=False,
                            pooling='max', input_shape=(224, 224, 3))
        logging.info("VGG16 model loaded successfully.")

        # Print the summary of the model's architecture (optional, good for debugging)
        # vgg16_model.summary()

        # Freeze the layers of the pre-trained model so they are not trained
        # This is essential when using a pre-trained model for feature extraction
        for model_layer in vgg16_model.layers:
            model_layer.trainable = False
        logging.info("VGG16 model layers frozen (set to non-trainable).")

        # Load and process the images
        first_image_pil = load_image(first_image_path)
        second_image_pil = load_image(second_image_path)

        # Get embeddings for both images
        first_image_vector = get_image_embeddings(first_image_pil, vgg16_model)
        second_image_vector = get_image_embeddings(second_image_pil, vgg16_model)

        # Calculate similarity score
        similarity_score = calculate_cosine_similarity_score(first_image_vector, second_image_vector)
        
        logging.info(f"Image similarity check completed. Score: {similarity_score:.4f}")
        return similarity_score

    except Exception as e:
        logging.critical(f"An error occurred during image similarity check: {e}", exc_info=True)
        # Re-raise the exception after logging, so calling code knows it failed
        raise


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy images for demonstration if they don't exist
    # In a real scenario, you would replace these with paths to your actual images.


    # Test Cases
    print("\n--- Running Similarity Checks ---")

    # Test 1: Highly similar images
    try:
        ref_path = 'data\adhar_1.jpeg'
        inference_path = 'data\pan_card_fake.jpg'
        score1 = perform_image_similarity_check(ref_path, inference_path)
        print(f"Similarity score between '{os.path.basename(ref_path)}' and '{os.path.basename(inference_path)}': {score1:.4f}")
    except Exception as e:
        print(f"Failed to check similarity (Test 1): {e}")
