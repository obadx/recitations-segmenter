from recitations_segmenter.train.process_data import to_huggingface_dataset

if __name__ == '__main__':
    recitations_file = './recitations.yml'
    ds = to_huggingface_dataset(recitations_file, base_dir='./data')
    print(ds)
