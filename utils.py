def get_max_length(dataset, key):
    values = sum([dataset[split][key] for split in ['train', 'dev', 'test']], start=[])
    return max(map(len, values))


def get_elapsed_time(start, end):
    seconds = int(end - start)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'
