from recitations_segmenter.segment import generate_time_stamps

if __name__ == '__main__':
    time_stamps = generate_time_stamps(
        4, max_duration_samples=1000, max_featrues_len=3)
    print(time_stamps)
