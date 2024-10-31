import re
from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    # Use sorted with a custom key to sort numerically
    subtitles_paths = sorted(glob(dataset_path + '/*.ass'), key=lambda x: int(re.search(r'(\d+)', x).group()))

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        # Read Lines
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[30:]  # Start from line 30 to skip metadata
            lines = [",".join(line.split(',')[9:]) for line in lines]

        # Remove \N and text enclosed in curly braces
        lines = [re.sub(r'\\N', ' ', line) for line in lines]
        lines = [re.sub(r'\{.*?\}', '', line) for line in lines]

        # Join lines to form the script for this file
        script = " ".join(lines)

        # Extract episode number from filename
        filename = path.split('\\')[-1]
        try:
            episode = int(re.search(r'(\d+)', filename).group())
        except ValueError:
            episode = None  # Handle invalid filenames gracefully

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame({"episode": episode_num, "script": scripts})
    return df
