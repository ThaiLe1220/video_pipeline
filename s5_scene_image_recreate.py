import argparse
import json
import os
import time
import shutil
import sys
import concurrent.futures
import threading
import traceback

# Import our utility module
import image_utils

# Global locks for thread safety
recreation_data_lock = threading.Lock()
temp_generations_lock = threading.Lock()
print_lock = threading.Lock()


# Thread-safe printing
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()  # Force output to display immediately


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Recreate scene images using replicate API"
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
        "-f",
        "--force",
        action="store_true",
        help="Force regeneration of existing images",
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
        help="Model to use for image generation (use -l to list available models)",
    )
    parser.add_argument(
        "-l",
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Maximum number of concurrent threads (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each scene processing (default: 300)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between variant generations (default: 0.2)",
    )
    return parser.parse_args()


def load_recreation_data(video_id):
    json_path = f"src/video/{video_id}/recreation.json"
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        safe_print(f"Error loading recreation data: {str(e)}")
        traceback.print_exc()
        return None


def save_recreation_data(video_id, data):
    json_path = f"src/video/{video_id}/recreation.json"
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        safe_print(f"Error saving recreation data: {str(e)}")
        traceback.print_exc()
        return False


def load_temp_generations(video_id):
    json_path = f"src/video/{video_id}/temp_generations.json"
    with temp_generations_lock:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)
        return {"generations": []}


def save_temp_generations(video_id, data):
    json_path = f"src/video/{video_id}/temp_generations.json"
    with temp_generations_lock:
        # Sort generations by scene_number first, then by timestamp
        data["generations"] = sorted(
            data["generations"], key=lambda x: (x["scene_number"], x["timestamp"])
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)


def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)


def process_single_scene(scene, video_id, force, variants, model, delay):
    """Process a single scene - this will be called by the thread pool"""
    scene_number = scene["scene_number"]

    try:
        # Check if image already exists and whether to force regeneration
        if scene["recreated_data"]["image_path"] and not force:
            safe_print(
                f"Scene {scene_number} already has an image. Skipping. Use -f to force regeneration."
            )
            return None

        safe_print(f"Processing scene {scene_number}...")

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
                # Use our utility function
                metadata = image_utils.generate_image(
                    description, temp_image_path, model
                )

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

                # Add small delay between generations (reduce from 1s to improve performance)
                time.sleep(delay)

            except Exception as e:
                safe_print(
                    f"Error generating variant {v+1} for scene {scene_number}: {str(e)}"
                )
                traceback.print_exc()

        # If we have at least one successful generation, use the first one as default
        if generated_variants:
            # Copy the first variant to the creation directory
            default_variant = generated_variants[0]
            final_image_path = f"src/video/{video_id}/creation/scene_{scene_number}.png"

            try:
                # Copy the file first (outside of any locks)
                shutil.copy2(default_variant["temp_path"], final_image_path)
                safe_print(f"Set default image for scene {scene_number} from variant 1")

                # Update recreation data in a separate, shorter transaction
                update_success = update_recreation_data(
                    video_id, scene_number, final_image_path
                )
                if update_success:
                    safe_print(
                        f"Successfully updated recreation.json for scene {scene_number}"
                    )
                else:
                    safe_print(
                        f"Failed to update recreation.json for scene {scene_number}"
                    )

                safe_print(f"Completed processing scene {scene_number}")
                return scene_number

            except Exception as e:
                safe_print(
                    f"Error setting default image for scene {scene_number}: {str(e)}"
                )
                traceback.print_exc()
                return None
        else:
            safe_print(f"No successful variants generated for scene {scene_number}")
            return None

    except Exception as e:
        safe_print(f"Unexpected error processing scene {scene_number}: {str(e)}")
        traceback.print_exc()
        return None


def update_recreation_data(video_id, scene_number, image_path):
    """Update recreation data with new image path - fixed to avoid deadlocks"""
    try:
        with recreation_data_lock:
            # Load the data directly (not using load_recreation_data to avoid nested locks)
            json_path = f"src/video/{video_id}/recreation.json"
            try:
                with open(json_path, "r") as f:
                    recreation_data = json.load(f)
            except Exception as e:
                safe_print(f"Error reading recreation JSON: {str(e)}")
                return False

            # Find the scene in the recreation data
            updated = False
            scene_numbers = []
            for s in recreation_data["recreations"]:
                scene_numbers.append(s["scene_number"])
                if s["scene_number"] == scene_number:
                    s["recreated_data"]["image_path"] = image_path
                    updated = True
                    break

            # Save the data if updated
            if updated:
                try:
                    with open(json_path, "w") as f:
                        json.dump(recreation_data, f, indent=2)
                    return True
                except Exception as e:
                    safe_print(f"Error writing recreation JSON: {str(e)}")
                    return False
            else:
                safe_print(
                    f"Warning: Scene {scene_number} not found in recreation data"
                )
                safe_print(f"Available scene numbers: {sorted(scene_numbers)}")
                return False
    except Exception as e:
        safe_print(f"Error updating recreation data for scene {scene_number}: {str(e)}")
        traceback.print_exc()
        return False


def process_scenes(
    video_id,
    start_scene,
    end_scene,
    force,
    variants,
    model,
    max_threads,
    timeout,
    delay,
):
    # Load recreation data
    recreation_data = load_recreation_data(video_id)
    if not recreation_data:
        safe_print(f"Error: Could not load recreation data for video {video_id}")
        return [], []

    # Ensure directories exist
    creation_dir = f"src/video/{video_id}/creation"
    temp_dir = f"src/video/{video_id}/temp"
    ensure_directory_exists(creation_dir)
    ensure_directory_exists(temp_dir)

    # Determine how many scenes to process
    scenes = recreation_data["recreations"]
    if end_scene is None or end_scene > len(scenes):
        end_scene = len(scenes)

    # Get the scenes we'll process
    scenes_to_process = scenes[start_scene - 1 : end_scene]
    total_scenes = len(scenes_to_process)

    safe_print(f"Processing scenes {start_scene} to {end_scene} for video {video_id}")
    safe_print(f"Using model: {model}")
    safe_print(f"Maximum concurrent threads: {max_threads}")
    safe_print(f"Total scenes to process: {total_scenes}")

    # Keep track of successful and failed scenes
    successful_scenes = []
    failed_scenes = []

    # Map to keep track of running futures
    running_futures = {}

    # Set up a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit initial batch of tasks (up to max_threads)
        initial_batch = min(max_threads, len(scenes_to_process))
        for scene in scenes_to_process[:initial_batch]:
            future = executor.submit(
                process_single_scene, scene, video_id, force, variants, model, delay
            )
            running_futures[future] = scene["scene_number"]
            safe_print(f"Submitted scene {scene['scene_number']} for processing")

        # Process scenes in a more controlled way
        scenes_left = scenes_to_process[initial_batch:]
        last_activity_time = time.time()

        # Process until all work is done
        while running_futures:
            try:
                # Wait for the next future to complete, with a shorter timeout
                done, not_done = concurrent.futures.wait(
                    running_futures.keys(),
                    timeout=min(timeout, 30),  # Use a shorter timeout for waiting
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Check if we're stuck (no progress for a long time)
                current_time = time.time()
                if not done:
                    if (
                        current_time - last_activity_time > 120
                    ):  # 2 minutes of no activity
                        safe_print(
                            "WARNING: No progress for 2 minutes, checking thread status..."
                        )
                        for future, scene_num in list(running_futures.items()):
                            if future.running():
                                safe_print(f"Scene {scene_num} is still running")
                            else:
                                safe_print(f"Scene {scene_num} is in pending state")

                            # Cancel long-running tasks
                            if (
                                current_time - last_activity_time > 300
                            ):  # 5 minutes of no activity
                                safe_print(f"Cancelling stuck scene {scene_num}")
                                future.cancel()
                                failed_scenes.append(scene_num)
                                running_futures.pop(future)

                        # If we have capacity and scenes left, force scheduling the next scene
                        if scenes_left and len(running_futures) < max_threads:
                            next_scene = scenes_left.pop(0)
                            new_future = executor.submit(
                                process_single_scene,
                                next_scene,
                                video_id,
                                force,
                                variants,
                                model,
                                delay,
                            )
                            running_futures[new_future] = next_scene["scene_number"]
                            safe_print(
                                f"Forced scheduling of scene {next_scene['scene_number']}"
                            )
                            last_activity_time = current_time
                else:
                    # Update the last activity time when we have progress
                    last_activity_time = time.time()

                # Process completed futures
                for future in done:
                    scene_number = running_futures.pop(future)

                    try:
                        # Get the result with a short timeout
                        result = future.result(timeout=5)
                        if result:
                            successful_scenes.append(scene_number)
                            safe_print(
                                f"✅ Confirmed completion of scene {scene_number}"
                            )
                        else:
                            failed_scenes.append(scene_number)
                            safe_print(f"❌ Scene {scene_number} failed to process")

                        # Submit a new task if there are scenes left
                        if scenes_left:
                            next_scene = scenes_left.pop(0)
                            new_future = executor.submit(
                                process_single_scene,
                                next_scene,
                                video_id,
                                force,
                                variants,
                                model,
                                delay,
                            )
                            running_futures[new_future] = next_scene["scene_number"]
                            safe_print(
                                f"Submitted scene {next_scene['scene_number']} for processing"
                            )

                    except concurrent.futures.TimeoutError:
                        safe_print(f"⏱️ Scene {scene_number} processing timed out")
                        failed_scenes.append(scene_number)
                    except Exception as e:
                        safe_print(
                            f"❌ Scene {scene_number} processing failed with error: {str(e)}"
                        )
                        traceback.print_exc()
                        failed_scenes.append(scene_number)

                # Check for timeouts in not_done futures
                current_time = time.time()
                timed_out_futures = []
                for future in not_done:
                    if future.running() and hasattr(future, "_start_time"):
                        future_runtime = current_time - future._start_time
                        if future_runtime > timeout:
                            scene_number = running_futures[future]
                            safe_print(
                                f"⏱️ Cancelling scene {scene_number} due to timeout ({future_runtime:.1f}s)"
                            )
                            future.cancel()
                            timed_out_futures.append(future)
                            failed_scenes.append(scene_number)

                # Remove timed out futures
                for future in timed_out_futures:
                    if future in running_futures:
                        running_futures.pop(future)

            except Exception as e:
                safe_print(f"Error in process_scenes main loop: {str(e)}")
                traceback.print_exc()
                time.sleep(1)  # Brief pause to avoid spinning

            # Give a short pause to avoid hammering the CPU
            time.sleep(0.1)

    # Report summary
    safe_print("\n=== Processing Summary ===")
    safe_print(f"Total scenes processed: {total_scenes}")
    safe_print(f"Successfully processed: {len(successful_scenes)}")
    if successful_scenes:
        safe_print(f"Successful scenes: {sorted(successful_scenes)}")
    safe_print(f"Failed to process: {len(failed_scenes)}")
    if failed_scenes:
        safe_print(f"Failed scenes: {sorted(failed_scenes)}")

    return successful_scenes, failed_scenes


def main():

    # Parse arguments
    args = parse_arguments()

    # Check if we're just listing models
    if args.list_models:
        image_utils.list_available_models()
        return

    # Validate model
    try:
        image_utils.validate_model(args.model)
    except ValueError as e:
        safe_print(f"Error: {str(e)}")
        image_utils.list_available_models()
        return

    # Record start time
    start_time = time.time()

    try:
        # Process scenes with threading
        successful, failed = process_scenes(
            args.video_id,
            args.start,
            args.end,
            args.force,
            args.variants,
            args.model,
            args.threads,
            args.timeout,
            args.delay,
        )

        # Record end time and display total runtime
        end_time = time.time()
        total_time = end_time - start_time

        # Final report
        safe_print(f"\n=== Final Report ===")
        safe_print(f"Video ID: {args.video_id}")
        safe_print(f"Model used: {args.model}")
        safe_print(f"Successfully processed {len(successful)} scenes")
        safe_print(f"Failed to process {len(failed)} scenes")
        safe_print(f"Total processing time: {total_time:.2f} seconds")
        safe_print(
            f"Average time per scene: {total_time/max(1, len(successful)):.2f} seconds"
        )
        safe_print("Done!")

    except Exception as e:
        safe_print(f"Error in main processing: {str(e)}")
        traceback.print_exc()
        safe_print("Process terminated with errors.")

    # Return control to terminal with a newline
    safe_print("")


if __name__ == "__main__":
    main()

"""
# Process scene 1 with default model using 4 threads
python s5_scene_image_recreate.py 7472159628353113366 -s 1 -e 1

# Process multiple scenes with 8 threads and shorter delay
python s5_scene_image_recreate.py 7472159628353113366 -s 1 -e 15 -v 1 -t 8 --delay 0.1

# List available models
python s5_scene_image_recreate.py 7472159628353113366 -l

# Force regeneration with specific model and maximum threads
python s5_scene_image_recreate.py 7472159628353113366 -s 1 -e 3 -f -m black-forest-labs/flux-dev -t 4
"""
