import os
import time
import json
from datetime import datetime
from google import genai
from google.genai import types, Client
from typing import Optional
from dotenv import load_dotenv


def upload_video_and_poll(client: "Client", video_file_path: str) -> Optional[str]:
    """
    Uploads a video file and polls its status until it becomes ACTIVE.
    Returns the video file object if ACTIVE, or None if it fails.
    """
    video_file = client.files.upload(file=video_file_path)

    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "ACTIVE":
        return video_file

    return None


def process_video(model: str = "gemini-2.0-flash", video_file_path: str = None):
    """
    Initializes the client, uploads the video, and generates content using the Gemini model.
    Returns the response from the content generation.
    """
    # Initialize the client with the API key from the environment
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Upload the video file and wait until it is ACTIVE
    video_file = upload_video_and_poll(client, video_file_path)
    if video_file is None:
        return None

    try:
        # Define the prompt with detailed scene description instructions
        prompt = (
            "Analyze this TikTok video and provide detailed descriptions of each distinct scene. IMPORTANT: Only consider it a new scene if there is a significant camera view change or cut to a different location/setting. "
            "Do NOT consider the following as new scenes: text overlay changes, caption changes, subject movements, facial expressions, or small actions within the same camera view. "
            "For each scene, provide precise timestamps (MM:SS.mmm format) that can be used to accurately cut the video into segments. "
            "Create rich, detailed descriptions for each scene that capture: "
            "- Visual elements (setting, colors, lighting, composition) "
            "- People present (appearance, expressions, positioning) "
            "- Actions and movements (what people or objects are doing) "
            "- Important objects or props "
            "- Atmosphere or mood of the scene "
            "- Camera techniques or transitions "
            "Structure your response in JSON format as follows:\n\n"
            "{\n"
            '  "scenes": [\n'
            "    {\n"
            '      "description": "Detailed description of what\'s happening in this scene.",\n'
            '      "time": "MM:SS.mmm",\n'
            '      "text_overlay": "Any text displayed on screen during this scene. Leave empty if no text overlay exists."\n'
            "    },\n"
            "    {\n"
            '      "description": "Another scene description.",\n'
            '      "time": "MM:SS.mmm",\n'
            '      "text_overlay": ""\n'
            "    }\n"
            "    // Add additional scenes as needed\n"
            "  ]\n"
            "}"
            "Return ONLY the JSON object with no additional text, explanation, or markdown formatting."
        )

        # Build the content with both the video file and the text prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri,
                        mime_type=video_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        # Set the generation configuration
        config = types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )

        # Generate content using the Gemini model
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    finally:
        # Delete the uploaded video file regardless of the generation outcome
        try:
            deletion_response = client.files.delete(name=video_file.name)
        except Exception as e:
            pass

    return response


def save_json_output(response_text, video_file_path):
    """
    Saves the JSON response to a file in the src/json directory.
    Handles responses that contain JSON within markdown code blocks.
    Skips saving if a file with the same name already exists.

    Args:
        response_text (str): The response text that might contain JSON
        video_file_path (str): Path to the original video file

    Returns:
        str: Path to the saved JSON file or None if skipped
    """
    # Create the src/json directory if it doesn't exist
    json_dir = "src/json"
    os.makedirs(json_dir, exist_ok=True)

    # Extract the video filename without extension
    video_filename = os.path.basename(video_file_path)
    video_name = os.path.splitext(video_filename)[0]

    # Create the output filename
    json_filename = f"{video_name}.json"
    json_path = os.path.join(json_dir, json_filename)

    # Check if file already exists
    if os.path.exists(json_path):
        print(f"File {json_path} already exists, skipping save")
        return json_path

    # Try to extract JSON from markdown code blocks if present
    import re

    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)

    if json_match:
        json_text = json_match.group(1)
    else:
        json_text = response_text.strip()

    # Clean the JSON text by removing any bad control characters
    json_text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", json_text)

    try:
        # Parse the cleaned text to ensure it's valid JSON
        json_data = json.loads(json_text)

        # Write the JSON to file with pretty formatting
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

        print(f"Successfully saved JSON to {json_path}")
        return json_path

    except json.JSONDecodeError as e:
        print(f"Error: Response is not valid JSON: {e}")

        # Save the raw response to a text file for debugging
        debug_path = f"{json_path}.txt"
        with open(debug_path, "w", encoding="utf-8") as debug_file:
            debug_file.write(response_text)
        print(f"Saved raw response to {debug_path} for debugging")

        return None


def main():
    """
    Main function to process video.mp4 file.
    Loads API key from .env file and processes the video.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Check if API key is available
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in .env file")
        return

    video_file_path = "src/video/7472159628353113366.mp4"

    # Check if video file exists
    if not os.path.exists(video_file_path):
        print(f"Error: Video file '{video_file_path}' not found")
        return

    # Process the video
    print(f"Processing video: {video_file_path}")

    response = process_video("gemini-2.0-flash", video_file_path)
    # response = process_video("gemini-2.0-flash-thinking-exp", video_file_path)
    # response = process_video("gemini-2.0-pro-exp", video_file_path)

    # Handle the response
    if response is None:
        print("Error: Failed to process the video")
    else:
        print("Video processed successfully")

        # Save the JSON output to a file
        json_file_path = save_json_output(response.text, video_file_path)
        print(f"JSON output saved to: {json_file_path}")

        # Also print the response for debugging purposes
        print("Response content:")
        print(response.text)


if __name__ == "__main__":
    main()
