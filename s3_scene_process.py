import os
import json
import argparse


def map_scenes_to_data(video_id):
    """
    Map scene images to their corresponding scene data from the JSON file

    Args:
        video_id (str): ID of the video to process
    """
    # Define file paths
    json_dir = os.path.join("src", "json")
    video_dir = os.path.join("src", "video")
    output_dir = os.path.join("src", "video", video_id)

    # Construct file paths
    json_path = os.path.join(json_dir, f"{video_id}.json")
    scenes_dir = os.path.join(video_dir, video_id, "original")
    output_json_path = os.path.join(output_dir, "scenes.json")

    # Check if files and directories exist
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    if not os.path.exists(scenes_dir):
        print(f"Error: Scenes directory not found: {scenes_dir}")
        print("Have you run the extract_frames script first?")
        return

    # Load JSON data
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        return

    # Check if scenes data exists
    if "scenes" not in data or not data["scenes"]:
        print("Error: No scenes found in JSON data")
        return

    # Get list of scene image files
    scene_files = [
        f
        for f in os.listdir(scenes_dir)
        if f.startswith("scene_") and f.endswith(".jpg")
    ]
    scene_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    if not scene_files:
        print(f"Error: No scene image files found in: {scenes_dir}")
        return

    print(
        f"Found {len(scene_files)} scene images and {len(data['scenes'])} scene data entries"
    )

    # Prepare output data
    mapped_scenes = []

    # Process all scenes
    for i in range(min(len(scene_files), len(data["scenes"]))):
        scene_file = scene_files[i]
        scene_data = data["scenes"][i]

        # Construct relative path
        relative_path = os.path.join(video_id, "original", scene_file)

        # Create mapping object
        scene_mapping = {
            "image_path": f"src/video/{relative_path}",
            "scene_data": scene_data,
        }

        # Add to list of mapped scenes
        mapped_scenes.append(scene_mapping)

        # Print the mapping
        print(f"\n====== Scene {i+1} Mapping ======")
        print(f"Image path: src/video/{relative_path}")
        print("Scene data:")
        print(json.dumps(scene_data, indent=2))

    # Save the mapped scenes to a new JSON file
    output_data = {
        "video_id": video_id,
        "scene_count": len(mapped_scenes),
        "mapped_scenes": mapped_scenes,
    }

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # Write the JSON file
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSuccessfully saved scene mappings to: {output_json_path}")
    except Exception as e:
        print(f"Error saving scene mappings file: {e}")


def main():
    """
    Main function to map scene images to their corresponding scene data
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Map scene images to their corresponding scene data"
    )
    parser.add_argument(
        "video_id", help="ID of the video to process (without extension)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the mapping
    map_scenes_to_data(args.video_id)


if __name__ == "__main__":
    main()

# python s3_scene_process.py 7472159628353113366
