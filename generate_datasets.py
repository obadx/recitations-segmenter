from recitations_segmenter.train.process_data import to_huggingface_dataset

if __name__ == '__main__':
    recitations_file = './recitations.yml'
    ds = to_huggingface_dataset(
        recitations_file, base_dir='./data', limit=512, batch_size=256)
    print(ds)

    for idx, item in enumerate(ds['recitation_0']):
        for k, v in item.items():
            print(f'{k}:\n{v}\n\n')
