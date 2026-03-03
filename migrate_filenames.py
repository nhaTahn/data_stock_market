import os
import glob
import re

def migrate_filenames(directory='data'):
    files = glob.glob(f"{directory}/**/*.csv", recursive=True)
    count = 0
    pattern = re.compile(r'^(.*?)_from_.*\.csv$')
    for filepath in files:
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        
        match = pattern.match(base_name)
        if match:
            symbol = match.group(1)
            new_name = f"{symbol}.csv"
            new_filepath = os.path.join(dir_name, new_name)
            
            # If the new file already exists, we might need to handle it or skip
            # Assuming it's safe to just rename and potentially overwrite if needed, or better, remove old
            if os.path.exists(new_filepath) and new_filepath != filepath:
                print(f"File {new_filepath} already exists. Removing older {filepath}")
                os.remove(filepath)
            else:
                os.rename(filepath, new_filepath)
                count += 1
                
    print(f"Renamed {count} files.")

if __name__ == '__main__':
    migrate_filenames()
