import soundfile as sf
from pathlib import Path
from tqdm import tqdm

data_dir = Path("data/0/000_versebyverse.zip")
for filepath in tqdm(data_dir.glob('*.mp3')):
    try:
        data, sr = sf.read(filepath)
        # print(f"✅ Success: {filepath}")
    except Exception as e:
        print(f"❌ Failed: {filepath} - Error: {str(e)}")
