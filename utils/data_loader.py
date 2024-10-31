from glob import glob 
import pandas as pd

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path+'/*.ass')

    scripts=[]
    episode_num=[]

    for path in subtitles_paths:

        #Read Lines
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[30:]
            lines =  [ ",".join(line.split(',')[9:])  for line in lines ]
        
        lines = [ line.replace('\\N',' ') for line in lines]
        script = " ".join(lines)

        filename = path.split('\\')[-1]  # Extracts the filename from the path
        try:
            episode = int(filename.split('-')[-1].split('.')[0].strip())
        except ValueError:
            episode = None  # Or some default value or logic to handle invalid filenames


        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode":episode_num, "script":scripts })
    return df