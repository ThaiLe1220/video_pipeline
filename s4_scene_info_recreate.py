import argparse
import base64
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process video scenes with Gemini AI")

    parser.add_argument("video_id", help="ID of the video to process")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force recreation of scenes even if they already exist",
    )
    parser.add_argument(
        "--start", "-s", type=int, default=1, help="Start scene number (1-indexed)"
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=1,
        help="End scene number (1-indexed, inclusive)",
    )
    parser.add_argument(
        "--additional",
        "-a",
        type=str,
        default="",
        help="Additional text to include in the Gemini prompt",
    )

    return parser.parse_args()


def load_scenes_data(video_id):
    """Load scenes data from the JSON file"""
    json_path = f"src/video/{video_id}/scenes.json"

    with open(json_path, "r") as file:
        video_data = json.load(file)

    return video_data


def load_existing_recreations(video_id):
    """Load existing recreation data if available"""
    output_path = f"src/video/{video_id}/recreation.json"

    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Existing recreation file is corrupted. Creating new file.")
            return {"video_id": video_id, "processed_scenes": 0, "recreations": []}

    return {"video_id": video_id, "processed_scenes": 0, "recreations": []}


def save_recreation_data(video_id, recreations):
    """Save recreation data to JSON file"""
    output_dir = f"src/video/{video_id}"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/recreation.json"

    with open(output_path, "w") as file:
        json.dump(recreations, file, indent=2)

    print(f"Recreation data saved to {output_path}")


def recreate_scene(client, scene_data, scene_number, additional_text=""):
    """Process a single scene with Gemini API"""
    try:
        image_path = scene_data["image_path"]
        description = scene_data["scene_data"]["description"]

        files = [
            client.files.upload(file=image_path),
        ]

        model = "gemini-2.0-flash"

        # Base prompt
        prompt_text = f"""Convert this scene and reference image into a JSON object with 'image_description' (MidJourney 6.1 prompt) and 'motion_description' (Kling video paragraph). Follow these specs:

- **Image Description**:
  • Create an EXPANSIVE, HIGHLY DETAILED prompt (150-200 words)
  • Extract ALL visual elements from reference image and scene description
  • Include iconic objects and background elements in detail
  • Describe 4-5 material textures + light interaction properties
  • Specify color gradients and atmospheric qualities
  • Maintain consistent perspective throughout

- **Motion Description**:
  • 60-word max description focusing on 1-2 CORE MOTIONS with HIGH INTENSITY
  • Emphasize DYNAMIC camera movements (unless scene requires stillness)
  • Use strong motion verbs that convey speed and impact
  • Focus on PRIMARY motion first, then secondary interactions
  • Create clear beginning → peak → resolution flow
  • Match motion style to scene's emotional tone

Scene Description: {description}"""

        # Add additional text if provided
        if additional_text:
            prompt_text += f"\n\nAdditional Instructions: {additional_text}"

        prompt_text += "\n\nRespond ONLY with valid JSON. No commentary."

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files[0].uri,
                        mime_type=files[0].mime_type,
                    ),
                    types.Part.from_text(text=prompt_text),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""```json
{
  \"image_description\": \"Captured from a first-person perspective, a serene scene unfolds. Bare feet rest on a polished marble ledge, adorned in flowing, ornate pants of shimmering gold and intricate white patterns. The fabric catches the sunlight, creating a soft, diffused glow, highlighting the detailed floral designs. Beyond the ledge, thick, luxurious brown curtains frame an otherworldly vista. Towering white structures rise majestically, their surfaces gleaming under the bright sunlight. Waterfalls cascade down tiered levels into a vibrant, azure body of water, its surface reflecting the partly cloudy sky. The architectural style is grandiose and fantastical, reminiscent of a lost city. Light plays across the scene, accentuating the textures of the marble, the flowing curtains, and the water's surface, creating a harmonious blend of natural and artificial elements. The overall atmosphere is one of peaceful luxury and impending doom, suggesting an ethereal paradise on the brink of destruction.\",
  \"motion_description\": \"The camera slowly pans up from the bare feet on the ledge, gradually revealing the panoramic view of Atlantis. Waterfalls cascade vigorously in the background as the camera continues its smooth ascent, showcasing the impending disaster. A distant structure crumbles as the shot concludes.\"
}
```"""
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_mime_type="application/json",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        # Clean the response to extract valid JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        # Parse JSON
        gemini_response = json.loads(response_text)

        # Create result with metadata and new structure
        result = {
            "scene_index": scene_number - 1,  # Store 0-indexed
            "scene_number": scene_number,  # Store 1-indexed
            "original_data": scene_data,
            "recreated_data": {
                "image_description": gemini_response.get("image_description", ""),
                "image_path": "",
                "video_description": gemini_response.get("motion_description", ""),
                "video_path": "",
            },
        }

        print(f"Successfully processed scene {scene_number}")
        return result

    except Exception as e:
        print(f"Error processing scene {scene_number}: {e}")
        return None


def process_video_scenes(
    video_id, start_scene=1, end_scene=1, force=False, additional_text=""
):
    """Process scenes from a video with multithreading support"""
    # Load environment variables from .env file
    load_dotenv()

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Load existing recreation data
    existing_data = load_existing_recreations(video_id)
    existing_scenes = {
        item["scene_number"]: item for item in existing_data.get("recreations", [])
    }

    # Load scenes data
    video_data = load_scenes_data(video_id)
    mapped_scenes = video_data.get("mapped_scenes", [])
    scene_count = video_data.get("scene_count", len(mapped_scenes))

    # Validate start and end scene numbers
    if start_scene < 1:
        start_scene = 1
    if end_scene > scene_count:
        end_scene = scene_count
    if start_scene > end_scene:
        print(
            f"Start scene {start_scene} is greater than end scene {end_scene}. Swapping."
        )
        start_scene, end_scene = end_scene, start_scene

    # Determine which scenes to process
    scenes_to_process = []
    for scene_num in range(start_scene, end_scene + 1):
        # Skip already processed scenes unless force flag is set
        if scene_num in existing_scenes and not force:
            print(f"Scene {scene_num} already processed. Skipping.")
            continue

        scene_idx = scene_num - 1  # Convert to 0-indexed
        if scene_idx < len(mapped_scenes):
            scenes_to_process.append((mapped_scenes[scene_idx], scene_num))

    if not scenes_to_process:
        print("No scenes to process. All requested scenes already exist.")
        return existing_data

    # Process scenes with ThreadPoolExecutor
    new_recreations = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_scene = {
            executor.submit(
                recreate_scene, client, scene_data, scene_num, additional_text
            ): scene_num
            for scene_data, scene_num in scenes_to_process
        }

        for future in as_completed(future_to_scene):
            scene_num = future_to_scene[future]
            try:
                result = future.result()
                if result:
                    new_recreations.append(result)
            except Exception as exc:
                print(f"Scene {scene_num} generated an exception: {exc}")

    # Sort new recreations by scene number to maintain order
    new_recreations.sort(key=lambda x: x["scene_number"])

    # Merge with existing recreations, replacing if needed
    merged_recreations = list(existing_data.get("recreations", []))

    # Remove any recreations that we're replacing
    merged_recreations = [
        r
        for r in merged_recreations
        if r["scene_number"] not in [nr["scene_number"] for nr in new_recreations]
    ]

    # Add new recreations
    merged_recreations.extend(new_recreations)

    # Sort all recreations by scene number
    merged_recreations.sort(key=lambda x: x["scene_number"])

    # Update and save results
    result = {
        "video_id": video_id,
        "processed_scenes": len(merged_recreations),
        "recreations": merged_recreations,
    }

    save_recreation_data(video_id, result)

    return result


if __name__ == "__main__":
    args = parse_arguments()

    process_video_scenes(
        args.video_id,
        start_scene=args.start,
        end_scene=args.end,
        force=args.force,
        additional_text=args.additional,
    )

"""
# Process scene 1 only
python s4_scene_info_recreate.py 7472159628353113366
python s4_scene_info_recreate.py 7472159628353113366 -f

# Process scenes
python s4_scene_info_recreate.py 7472159628353113366 -s 1 -e 15

# Force reprocessing with additional instructions
python s4_scene_info_recreate.py 7472159628353113366 -f -a "Emphasize the doomed nature of Atlantis"
"""
