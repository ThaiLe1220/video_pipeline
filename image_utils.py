import os
import time
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the API key for the replicate client
api_key = os.getenv("REPLICATE_API_KEY")

if not api_key:
    print("Error: REPLICATE_API_KEY not found in environment variables.")

os.environ["REPLICATE_API_TOKEN"] = api_key

# Define available models with details and default configurations
AVAILABLE_MODELS = {
    "black-forest-labs/flux-schnell": {
        "description": "The fastest image generation model tailored for local development and personal use",
        "price": "$0.003 / image",
        "default": True,
        "config": {
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "9:16",
            "output_format": "png",
            "output_quality": 80,
            "num_inference_steps": 4,
        },
    },
    "black-forest-labs/flux-dev": {
        "description": "A 12 billion parameter rectified flow transformer capable of generating images from text descriptions",
        "price": "$0.0025 / image",
        "default": False,
        "config": {
            "go_fast": True,
            "guidance": 3.5,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "9:16",
            "output_format": "png",
            "output_quality": 80,
            "prompt_strength": 0.8,
            "num_inference_steps": 28,
        },
    },
    "black-forest-labs/flux-dev-lora": {
        "description": "A version of flux-dev, a text to image model, that supports fast fine-tuned lora inference",
        "price": "$0.032 / image",
        "default": False,
        "config": {
            "go_fast": True,
            "guidance": 3,
            "lora_scale": 1,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "9:16",
            # "lora_weights": "huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur",
            "lora_weights": "huggingface.co/strangerzonehf/Flux-Midjourney-Mix2-LoRA",
            "output_format": "png",
            "output_quality": 80,
            "prompt_strength": 0.8,
            "num_inference_steps": 28,
        },
    },
    "black-forest-labs/flux-1.1-pro": {
        "description": "Faster, better FLUX Pro. Text-to-image model with excellent image quality, prompt adherence, and output diversity",
        "price": "$0.04 / image",
        "default": False,
        "config": {
            "aspect_ratio": "9:16",
            "output_format": "png",
            "output_quality": 80,
            "safety_tolerance": 2,
            "prompt_upsampling": True,
        },
    },
}


def list_available_models():
    """Print available models with their descriptions and pricing"""
    print("Available models:")
    for model_name, details in AVAILABLE_MODELS.items():
        default_text = " (default)" if details["default"] else ""
        print(f"- {model_name}{default_text}")
        print(f"  Description: {details['description']}")
        print(f"  Price: {details['price']}")
        print()


def check_api_key():
    """Check if the Replicate API key is set in the environment"""
    api_key = os.environ.get("REPLICATE_API_TOKEN")
    if not api_key:
        raise EnvironmentError("REPLICATE_API_TOKEN not set in environment variables")
    return api_key


def validate_model(model):
    """Validate if the model is in the list of available models"""
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model}' is not available. Use list_available_models() to see available models."
        )
    return True


def generate_image(
    prompt,
    output_path=None,
    model="black-forest-labs/flux-schnell",
    return_data=False,
    custom_config=None,
):
    """
    Generate an image using the specified model and prompt.

    Args:
        prompt (str): The text prompt for image generation
        output_path (str, optional): Path to save the generated image
        model (str): Model to use for generation (must be in AVAILABLE_MODELS)
        return_data (bool): Whether to return the image data
        custom_config (dict, optional): Custom configuration parameters to override defaults

    Returns:
        dict: Metadata about the generation
    """
    # Check API key
    check_api_key()

    # Validate model
    validate_model(model)

    # Always use the specific model's config
    config = AVAILABLE_MODELS[model]["config"].copy()

    # Add prompt to config
    config["prompt"] = prompt

    # Only override with custom config if explicitly provided
    if custom_config:
        config.update(custom_config)

    # Generate the image
    output = replicate.run(model, input=config)

    image_data = None

    # Process the output
    for item in output:
        if output_path:
            with open(output_path, "wb") as file:
                if return_data:
                    # If we need to return data, read it first
                    image_data = item.read()
                    file.write(image_data)
                else:
                    # Otherwise just write directly
                    file.write(item.read())
        elif return_data:
            # If no output path but return_data is True
            image_data = item.read()
        break  # Just use the first output

    # Prepare metadata
    metadata = {
        "model": model,
        "config": config,
        "timestamp": int(time.time()),
    }

    # Add image data if requested
    if return_data:
        metadata["image_data"] = image_data

    return metadata
