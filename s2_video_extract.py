import os
import json
import cv2
import argparse
import time


def time_to_milliseconds(time_str):
    """
    Convert time string in format MM:SS.mmm to milliseconds

    Args:
        time_str (str): Time in format "MM:SS.mmm"

    Returns:
        int: Time in milliseconds
    """
    # Handle potential formatting issues
    if not time_str or ":" not in time_str:
        return 0

    try:
        # Split time string into minutes, seconds (and milliseconds)
        parts = time_str.split(":")
        minutes = int(parts[0])

        # Handle seconds part which may include milliseconds
        if "." in parts[1]:
            sec_parts = parts[1].split(".")
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1])
        else:
            seconds = int(parts[1])
            milliseconds = 0

        # Convert to total milliseconds
        total_ms = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
        return total_ms

    except (ValueError, IndexError):
        print(f"Error parsing time string: {time_str}")
        return 0


def extract_frames(video_id, time_offset=2, width=576):
    # Start timing
    start_time = time.time()
    """
    Extract frames from a specific video at timestamps specified in its JSON file
    Adds specified time offset to each timestamp and uses simplified scene numbering
    Resizes frames to specified width while maintaining aspect ratio

    Args:
        video_id (str): ID of the video to process
        time_offset (float): Number of seconds to add to each timestamp (default: 2)
        width (int): Target width for output frames (default: 576)
    """
    # Convert time offset to milliseconds
    offset_ms = int(time_offset * 1000)

    # Define file paths
    json_dir = os.path.join("src", "json")
    video_dir = os.path.join("src", "video")

    # Construct file paths
    json_path = os.path.join(json_dir, f"{video_id}.json")

    # Find the video file
    video_path = None
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        potential_path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(potential_path):
            video_path = potential_path
            break

    # Check if files exist
    if not video_path:
        print(f"Error: Video file not found for ID: {video_id}")
        return

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    # Load JSON data
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        return

    # Create output directory
    output_dir = os.path.join("src", "video", video_id, "original")
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = (total_frames / fps) * 1000

    print(f"Video loaded: {os.path.basename(video_path)}")
    print(
        f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration_ms/1000:.2f} seconds"
    )
    print(f"Using time offset: +{time_offset} seconds")
    print(f"Output width: {width}px")

    # Process each scene
    if "scenes" not in data:
        print("Error: No 'scenes' field in JSON data")
        return

    for i, scene in enumerate(data["scenes"]):
        if "time" not in scene:
            print(f"Warning: Scene {i+1} has no timestamp, skipping")
            continue

        # Convert time to milliseconds and add offset
        time_ms = time_to_milliseconds(scene["time"]) + offset_ms

        # Make sure we don't exceed the video duration
        if time_ms >= duration_ms:
            print(
                f"Warning: Scene {i+1} timestamp (+{time_offset}s) exceeds video length, using last frame instead"
            )
            time_ms = duration_ms - 100  # 100ms before the end

        # Convert time to frame number
        frame_number = int((time_ms / 1000) * fps)

        # Set video position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame for scene {i+1}, skipping")
            continue

        # Resize the frame to target width while maintaining aspect ratio
        height, original_width = frame.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_frame = cv2.resize(
            frame, (width, new_height), interpolation=cv2.INTER_AREA
        )

        # Create simplified frame filename
        frame_filename = f"scene_{i+1}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        # Save the frame
        cv2.imwrite(frame_path, resized_frame)
        print(
            f"Saved frame for scene {i+1} at time +{time_offset}s: {time_ms/1000:.2f}s to {frame_path} ({width}x{new_height})"
        )

    # Release video capture
    cap.release()

    # Calculate and print processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(
        f"Completed extraction of {len(data['scenes'])} scenes from video ID: {video_id}"
    )
    print(f"Total processing time: {processing_time:.2f} seconds")


def main():
    """
    Main function to extract frames from a single video based on JSON scene information
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract frames from a video based on scene timestamps"
    )
    parser.add_argument(
        "video_id", help="ID of the video to process (without extension)"
    )
    parser.add_argument(
        "--offset",
        "-o",
        type=float,
        default=2.0,
        help="Time offset in seconds to add to each timestamp (default: 2.0)",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=576,
        choices=[576, 720],
        help="Output frame width in pixels (choices: 576, 720, default: 576)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the specified video with the given time offset and width
    extract_frames(args.video_id, args.offset, args.width)


if __name__ == "__main__":
    main()

# Example usage:
# python s2_video_extract.py 7472159628353113366 -o 2.5
# python s2_video_extract.py 7472159628353113366 -o 2.5 -w 720
