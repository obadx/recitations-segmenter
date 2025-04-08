from datasets import load_dataset
import json

if __name__ == '__main__':
    ds_path = '/cluster/users/shams035u1/data/segmentation-datasets/recitation-segmentation-augmented'
    ds_dict = load_dataset(ds_path)

    durations = []
    for split in ds_dict:
        durations += ds_dict[split]['duration']

    with open('durations.json', 'w') as f:
        json.dump(durations, f)
