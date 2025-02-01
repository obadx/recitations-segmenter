from pathlib import Path

from recitations_segmenter.utils import get_audiofiles, SURA_TO_AYA_COUNT


def cond_callback(p: Path) -> bool:
    name = p.name.split('.')[0]
    if name in ['audhubillah', 'bismillah']:
        return True
    try:
        if len(name) != 6:
            return False
        int_name = int(name)
        sura_idx = int_name // 1000
        aya_name = int_name % 1000
        if sura_idx >= 1 and sura_idx <= 114:
            # 0 idx represnet bismillah
            if aya_name <= SURA_TO_AYA_COUNT[sura_idx]:
                return True
    except Exception as e:
        pass

    return False


if __name__ == '__main__':
    # path = Path('../prepare-quran-dataset/frontend/quran-dataset')
    path = Path('./data/1')
    pathes = get_audiofiles(path, cond_callback)

    for p in pathes:
        print(p.absolute())

    print(len(pathes))
