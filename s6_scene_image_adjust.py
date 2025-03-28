import argparse
import json
import os
import shutil
import time
import threading
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import the image utils module
import image_utils

# Global locks for thread safety
temp_generations_lock = threading.Lock()
print_lock = threading.Lock()


# Thread-safe printing
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()  # Force output to display immediately


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="List and adjust scene images for videos"
    )
    parser.add_argument("video_id", help="ID of the video to process")
    parser.add_argument(
        "-s", "--start", type=int, default=1, help="Start scene number (default: 1)"
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=None,
        help="End scene number (default: all scenes)",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all temporary images for the specified scenes",
    )
    parser.add_argument(
        "-u",
        "--update",
        type=str,
        help="Update the chosen image with the specified temporary image path",
    )
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="Generate new temporary images for the specified scenes",
    )
    parser.add_argument(
        "-v",
        "--variants",
        type=int,
        default=1,
        help="Number of variants to generate per scene (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="black-forest-labs/flux-schnell",
        help="Model to use for image generation",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Maximum number of concurrent threads (default: 4)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between variant generations (default: 0.5)",
    )
    return parser.parse_args()


def load_recreation_data(video_id):
    json_path = f"src/video/{video_id}/recreation.json"
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        safe_print(f"Error loading recreation data: {str(e)}")
        return None


def save_recreation_data(video_id, data):
    json_path = f"src/video/{video_id}/recreation.json"
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        safe_print(f"Error saving recreation data: {str(e)}")
        return False


def load_temp_generations(video_id):
    json_path = f"src/video/{video_id}/temp_generations.json"
    with temp_generations_lock:  # Added lock for thread safety
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
        return {"generations": []}


def save_temp_generations(video_id, data):
    json_path = f"src/video/{video_id}/temp_generations.json"
    with temp_generations_lock:  # Added lock for thread safety
        # Sort generations by scene_number first, then by timestamp
        data["generations"] = sorted(
            data["generations"], key=lambda x: (x["scene_number"], x["timestamp"])
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)


def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)


def list_temp_images(video_id, start_scene, end_scene):
    # Load temp generations data
    temp_generations = load_temp_generations(video_id)

    # If no temp generations found, return
    if not temp_generations["generations"]:
        safe_print("No temporary generations found.")
        return

    # Filter generations by scene range
    filtered_generations = [
        gen
        for gen in temp_generations["generations"]
        if start_scene <= gen["scene_number"] <= end_scene
    ]

    if not filtered_generations:
        safe_print(
            f"No temporary generations found for scenes {start_scene} to {end_scene}."
        )
        return

    # Group generations by scene number
    scene_generations = {}
    for gen in filtered_generations:
        scene_num = gen["scene_number"]
        if scene_num not in scene_generations:
            scene_generations[scene_num] = []
        scene_generations[scene_num].append(gen)

    # Print organized results
    safe_print(f"\nTemporary images for scenes {start_scene} to {end_scene}:")
    safe_print("-" * 60)
    for scene_num in sorted(scene_generations.keys()):
        safe_print(f"\nScene {scene_num}:")
        for i, gen in enumerate(scene_generations[scene_num], 1):
            # Extract just the filename from the path for cleaner output
            filename = os.path.basename(gen["temp_path"])
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(gen["timestamp"]))
            safe_print(f"  {i}. {gen['temp_path']} (Generated: {ts})")


def update_scene_image(video_id, temp_image_path, start_scene, end_scene):
    # Check if the temp image exists
    if not os.path.exists(temp_image_path):
        safe_print(f"Error: Temporary image not found at {temp_image_path}")
        return False

    # Extract scene number from the filename
    # Expected format: src/video/{video_id}/temp/scene_{scene_number}_{timestamp}.png
    filename = os.path.basename(temp_image_path)
    try:
        # Parse the scene number from the filename
        parts = filename.split("_")
        if len(parts) < 3 or parts[0] != "scene":
            raise ValueError("Invalid filename format")
        scene_number = int(parts[1])
    except (ValueError, IndexError):
        safe_print(
            f"Error: Could not determine scene number from the filename {filename}"
        )
        return False

    # Check if scene number is within the specified range
    if scene_number < start_scene or scene_number > end_scene:
        safe_print(
            f"Error: Scene number {scene_number} is outside the specified range ({start_scene}-{end_scene})"
        )
        return False

    # Load recreation data
    recreation_data = load_recreation_data(video_id)
    if not recreation_data:
        safe_print(f"Error: Could not load recreation data for video {video_id}")
        return False

    # Find the corresponding scene in the data
    scene_found = False
    for scene in recreation_data["recreations"]:
        if scene["scene_number"] == scene_number:
            scene_found = True
            break

    if not scene_found:
        safe_print(f"Error: Scene {scene_number} not found in recreation data")
        return False

    # Create the final image path in the creation directory
    creation_dir = f"src/video/{video_id}/creation"
    ensure_directory_exists(creation_dir)
    final_image_path = f"{creation_dir}/scene_{scene_number}.png"

    # Copy the temporary image to the creation directory
    try:
        shutil.copy2(temp_image_path, final_image_path)
        safe_print(f"Successfully updated image for scene {scene_number}:")
        safe_print(f"  Source: {temp_image_path}")
        safe_print(f"  Destination: {final_image_path}")

        # Update the JSON with the new image path
        for scene in recreation_data["recreations"]:
            if scene["scene_number"] == scene_number:
                scene["recreated_data"]["image_path"] = final_image_path
                break

        if save_recreation_data(video_id, recreation_data):
            safe_print(f"Recreation data updated for scene {scene_number}")
            return True
        else:
            safe_print(
                f"Error: Failed to save recreation data for scene {scene_number}"
            )
            return False
    except Exception as e:
        safe_print(f"Error updating image for scene {scene_number}: {str(e)}")
        return False


def process_single_scene(scene, video_id, variants, model, delay):
    """Process a single scene - this will be called by the thread pool"""
    scene_number = scene["scene_number"]

    safe_print(f"Processing scene {scene_number}...")

    # Ensure temp directory exists
    temp_dir = f"src/video/{video_id}/temp"
    ensure_directory_exists(temp_dir)

    # Generate variants
    generated_variants = []
    for v in range(variants):
        timestamp = int(time.time())
        temp_image_path = (
            f"src/video/{video_id}/temp/scene_{scene_number}_{timestamp}.png"
        )

        # Generate the image using the description
        description = scene["recreated_data"]["image_description"]
        try:
            safe_print(f"Generating variant {v+1} for scene {scene_number}...")
            # Use image_utils module
            metadata = image_utils.generate_image(description, temp_image_path, model)

            # Create generation info
            generation_info = {
                "scene_number": scene_number,
                "temp_path": temp_image_path,
                "timestamp": timestamp,
                "metadata": metadata,
                "prompt": description,
            }

            # Thread-safe update of temp_generations
            temp_generations = load_temp_generations(video_id)
            temp_generations["generations"].append(generation_info)
            save_temp_generations(video_id, temp_generations)

            generated_variants.append(generation_info)

            # Add configurable delay between generations
            time.sleep(delay)

        except Exception as e:
            safe_print(
                f"Error generating variant {v+1} for scene {scene_number}: {str(e)}"
            )

    return scene_number if generated_variants else None


def generate_temp_images(
    video_id, start_scene, end_scene, variants, model, max_threads, delay
):
    """Generate new temporary images for the specified scene range using multithreading"""
    # Load recreation data
    recreation_data = load_recreation_data(video_id)
    if not recreation_data:
        safe_print(f"Error: Could not load recreation data for video {video_id}")
        return

    # Filter scenes by the specified range
    scenes_to_process = [
        scene
        for scene in recreation_data["recreations"]
        if start_scene <= scene["scene_number"] <= end_scene
    ]

    if not scenes_to_process:
        safe_print(f"No scenes found in the range {start_scene} to {end_scene}.")
        return

    # Validate model before proceeding
    try:
        image_utils.validate_model(model)
    except ValueError as e:
        safe_print(f"Error: {str(e)}")
        safe_print(
            "You can use --list-models option in the s5_scene_image_recreate.py script to see available models."
        )
        return

    safe_print(
        f"Generating {variants} variants for {len(scenes_to_process)} scenes using model: {model}"
    )
    safe_print(
        f"Using {max_threads} concurrent threads with {delay}s delay between variants"
    )

    # Process scenes using a thread pool
    successful_scenes = []
    failed_scenes = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit each scene to the thread pool
        futures = {
            executor.submit(
                process_single_scene, scene, video_id, variants, model, delay
            ): scene["scene_number"]
            for scene in scenes_to_process
        }

        # Wait for all tasks to complete
        for future in futures:
            scene_number = futures[future]
            try:
                result = future.result()
                if result:
                    safe_print(
                        f"✅ Completed generating variants for scene {scene_number}"
                    )
                    successful_scenes.append(scene_number)
                else:
                    safe_print(
                        f"❌ Failed to generate variants for scene {scene_number}"
                    )
                    failed_scenes.append(scene_number)
            except Exception as e:
                safe_print(f"❌ Error processing scene {scene_number}: {str(e)}")
                failed_scenes.append(scene_number)

    # Print summary
    safe_print("\n=== Generation Summary ===")
    safe_print(f"Total scenes processed: {len(scenes_to_process)}")
    safe_print(f"Successfully generated: {len(successful_scenes)}")
    if successful_scenes:
        safe_print(f"Successful scenes: {sorted(successful_scenes)}")
    safe_print(f"Failed to generate: {len(failed_scenes)}")
    if failed_scenes:
        safe_print(f"Failed scenes: {sorted(failed_scenes)}")


def main():
    # Parse arguments
    args = parse_arguments()

    try:
        # Get video ID
        video_id = args.video_id

        # Load recreation data
        recreation_data = load_recreation_data(video_id)
        if not recreation_data:
            safe_print(f"Error: Recreation data not found for video {video_id}")
            return

        # Determine start and end scenes
        start_scene = args.start

        # If end scene is not specified, find max scene number
        if args.end is None:
            end_scene = max(
                scene["scene_number"] for scene in recreation_data["recreations"]
            )
        else:
            end_scene = args.end

        # Validate start and end scene
        if start_scene > end_scene:
            safe_print(
                f"Error: Start scene ({start_scene}) cannot be greater than end scene ({end_scene})"
            )
            return

        safe_print(f"Processing video {video_id}, scenes {start_scene} to {end_scene}")

        # Process commands
        if args.list:
            # List temporary images for the specified scene range
            list_temp_images(video_id, start_scene, end_scene)
        elif args.update:
            # Update chosen image with specified temporary image
            update_scene_image(video_id, args.update, start_scene, end_scene)
        elif args.generate:
            # Generate new temporary images
            generate_temp_images(
                video_id,
                start_scene,
                end_scene,
                args.variants,
                args.model,
                args.threads,
                args.delay,
            )
        else:
            safe_print(
                "No action specified. Use --list to list temporary images, --update to update a scene image, or --generate to create new temporary images."
            )
            safe_print("Examples:")
            safe_print(f"  python s6_scene_image_adjust.py {video_id} --list")
            safe_print(f"  python s6_scene_image_adjust.py {video_id} -s 1 -e 3 --list")
            safe_print(
                f"  python s6_scene_image_adjust.py {video_id} --update src/video/{video_id}/temp/scene_1_12345678.png"
            )
            safe_print(
                f"  python s6_scene_image_adjust.py {video_id} -s 1 -e 3 --generate -v 4"
            )
            safe_print(
                f'  python s6_scene_image_adjust.py {video_id} -s 1 -e 3 --generate -m "black-forest-labs/flux-1.1-pro"'
            )
            safe_print(
                f"  python s6_scene_image_adjust.py {video_id} -s 1 -e 3 --generate -t 8 --delay 0.2"
            )

    except Exception as e:
        safe_print(f"Error in main processing: {str(e)}")
        import traceback

        traceback.print_exc()
        safe_print("Process terminated with errors.")


if __name__ == "__main__":
    main()

"""
# Typical workflow example:

# 1. Generate new images
python s6_scene_image_adjust.py 7472159628353113366 -s 2 -e 2 -g

python s6_scene_image_adjust.py 7472159628353113366 -s 1 -e 1 -g -m "black-forest-labs/flux-dev"
python s6_scene_image_adjust.py 7472159628353113366 -s 1 -e 1 -g -m "black-forest-labs/flux-dev-lora"

# 2. List the generated images to choose from
python s6_scene_image_adjust.py 7472159628353113366 -s 2 -e 2 -l

# 3. Update with your preferred image
python s6_scene_image_adjust.py 7472159628353113366 -u src/video/7472159628353113366/temp/scene_2_1712345679.png

# List all generated images regardless of scene range
python s6_scene_image_adjust.py 7472159628353113366 -s 1 -e 20 -l

---

# Generate 2 variants using the high-quality flux-1.1-pro model for scenes 2-5
python s6_scene_image_adjust.py 7472159628353113366 -s 2 -e 5 -g -v 2 -m "black-forest-labs/flux-1.1-pro"

# Generate 4 variants with flux-dev for a single scene
python s6_scene_image_adjust.py 7472159628353113366 -s 3 -e 3 -g -v 4 -m "black-forest-labs/flux-dev"
"""
