from recitations_segmenter.train.process_data import to_huggingface_dataset

if __name__ == '__main__':
    recitations_file = './recitations.yml'
    ds = to_huggingface_dataset(
        recitations_file, base_dir='./data', num_proc=8, limit=10)
    print(ds)

    for k, v in ds['recitation_0'][0].items():
        print(f'{k}:\n{v}\n\n')
