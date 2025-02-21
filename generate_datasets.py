from recitations_segmenter.train.process_data import to_huggingface_16k_dataset, save_to_disk

if __name__ == '__main__':
    downlaod_dir = './data'
    recitations_file = './recitations.yml'
    out_path = '../segment-ds.hf'
    samples_per_shard = 128

    ds = to_huggingface_16k_dataset(recitations_file, base_dir=downlaod_dir)
    print(ds)

    save_to_disk(ds, out_path=out_path, samples_per_shard=samples_per_shard)
