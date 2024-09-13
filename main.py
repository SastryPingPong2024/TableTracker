import os
import argparse
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from process import process_video  # Import the process_video function

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to integers
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def set_available_gpu():
    try:
        gpu_memory_map = get_gpu_memory_map()
        if gpu_memory_map:
            # Select the GPU with the most free memory
            selected_gpu = gpu_memory_map.index(max(gpu_memory_map))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            return selected_gpu
        else:
            print("No GPUs available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi is not available. Make sure CUDA is installed and you have NVIDIA GPUs.")

def process_match(match_folder, root_dir):
    match_number = match_folder.split('match')[1]
    input_dir = os.path.join(root_dir, match_folder)
    output_dir = os.path.join('outputs', match_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the CUDA_VISIBLE_DEVICES environment variable
    gpu_id = set_available_gpu()

    # Process all videos in the match folder
    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Add or remove video extensions as needed
            video_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + '_table')
            print(f"Processing {video_file} from {match_folder} on GPU {gpu_id}")
            process_video(video_path, output_path)
            print(f"Finished processing {video_file} from {match_folder}")

def main(root_dir):
    match_folders = [f for f in os.listdir(root_dir) if f.startswith('match')]
    match_folders.sort(key=lambda x: int(x.split('match')[1]))

    with ProcessPoolExecutor(max_workers=28) as executor:
        futures = []
        for i, match_folder in enumerate(match_folders):
            future = executor.submit(process_match, match_folder, root_dir)
            futures.append(future)
            time.sleep(10)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in match folders in parallel using multiple GPUs")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    args = parser.parse_args()

    main(args.root_dir)