import requests
from pathlib import Path
import os
from zipfile import ZipFile, is_zipfile
from urllib.parse import urlparse
import re
import urllib
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
import json
from typing import Callable
import yaml


from pypdl import Pypdl
import filetype
from mutagen import File

DATA_PATH = Path(__file__).parent / 'data'


def overwrite_readme_yaml(file_path, metadata: dict | list):
    """Overwrite the metadata section (yaml) section of README.md file with new `metadata`
    """
    # Read the file
    lines = []
    if Path(file_path).is_file():
        with open(file_path, 'r') as f:
            lines = f.readlines()

    yaml_content = None
    rest_content = []
    has_yaml = False
    parser_idx = 0

    # Check if the file starts with YAML front matter
    if len(lines) > 0 and lines[0].strip() == '---':
        yaml_lines = []
        parser_idx = 1
        # Collect lines until the next '---'
        while parser_idx < len(lines) and lines[parser_idx].strip() != '---':
            yaml_lines.append(lines[parser_idx])
            parser_idx += 1
        # Check if closing '---' was found
        if parser_idx < len(lines) and lines[parser_idx].strip() == '---':
            has_yaml = True
            parser_idx += 1
            yaml_content = ''.join(yaml_lines)

    # If YAML exists, parse and edit it
    # if has_yaml:
    #     data = yaml.safe_load(yaml_content) or {}  # Handle empty YAML
    #     edit_callback(data)  # Apply user-defined edits

    rest_content = ''.join(lines[parser_idx:])  # Content after YAML block

    # Convert back to YAML
    new_content = rest_content
    if metadata:
        new_yaml = yaml.dump(metadata, default_flow_style=False)
        if not new_yaml.endswith('\n'):
            new_yaml += '\n'  # Ensure trailing newline

        # Reconstruct the file content
        new_content = f"---\n{new_yaml}---\n{rest_content}"

    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)


def get_suar_list(suar_path=DATA_PATH / 'suar_list.json') -> list[str]:
    """Return the suar names of the Holy Quran in an ordered list
    """
    with open(suar_path, 'r', encoding='utf8') as f:
        suar_list = json.load(f)
    return suar_list


def get_sura_to_aya_count(path=DATA_PATH / 'sura_to_aya_count.json') -> dict[int, int]:
    """Loads sura to aya count
    """
    with open(path, 'r') as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


SUAR_LIST = get_suar_list()
SURA_TO_AYA_COUNT = get_sura_to_aya_count()


def load_jsonl(filepath: str | Path) -> list[Any]:
    """Loads a Json file data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: list[Any], filepath: str | Path) -> None:
    """Saves list[Any] data into a `filepath` ins JSON line format"""
    data_str = ""
    for item in data:
        data_str += json.dumps(item, ensure_ascii=False) + '\n'
    if data_str:
        data_str = data_str[:-1]  # removes '\n' from last line

    with open(filepath, 'w+', encoding='utf-8') as f:
        f.write(data_str)


def get_audiofiles(
    path: Path,
    condition_callback: Callable[[Path], bool] = None,
    delete_audiofile_on_false_cond: bool = False,
) -> list[Path]:
    """retrive audio files from tree structerd files

    Args:
        condition_collback (Callable[[Path], bool]): an optinal conditional callaback to chose the audio file
        delete_audio_fole_on_false_cond (bool): delete the audiofile if it doesnot satisify the condition of `condition_callbak`
            * It will be use only if `condition_callback` is not None

    Example:
    .
    ├── 001.mp3
    ├── alhossary
    ├── hafs
    │   └── quran-recitations
    │       ├── 001.mp3
    │       ├── 008 - الانقال.mp3
    │       ├── 018.mp3
    │       ├── 053_مشاري.mp3
    │       ├── 054_الحصري.mp3
    │       ├── 054_الرزيقي.mp3
    │       ├── 054_محمود_عبدالحكم.mp3
    │       ├── 114.mp3
    │       └── elhossary_fast
    │           ├── 001 - الفاتحة.mp3
    │           ├── 008 - الانقال.mp3
    │           ├── 035 - فاطر.mp3
    │           ├── 048 - الفتح.mp3
    │           ├── 053 - النجم.mp3
    │           ├── 054 - القمر.mp3
    │           ├── 055 - الرحمن.mp3
    │           ├── 062 - الجمعة.mp3
    │           ├── 078 - النبأ.mp3
    │           ├── 093 - الضحى.mp3
    │           └── 094 - الشرح.mp3
    ├── reciters.jsonl
    └── test.zip

    So the output audio files:
        data/hafs/quran-recitations/054_الرزيقي.mp3
        data/hafs/quran-recitations/008 - الانقال.mp3
        data/hafs/quran-recitations/018.mp3
        data/hafs/quran-recitations/001.mp3
        data/hafs/quran-recitations/054_الحصري.mp3
        data/hafs/quran-recitations/elhossary_fast/093 - الضحى.mp3
        data/hafs/quran-recitations/elhossary_fast/055 - الرحمن.mp3
        data/hafs/quran-recitations/elhossary_fast/094 - الشرح.mp3
        data/hafs/quran-recitations/elhossary_fast/008 - الانقال.mp3
        data/hafs/quran-recitations/elhossary_fast/054 - القمر.mp3
        data/hafs/quran-recitations/elhossary_fast/062 - الجمعة.mp3
        data/hafs/quran-recitations/elhossary_fast/048 - الفتح.mp3
        data/hafs/quran-recitations/elhossary_fast/053 - النجم.mp3
        data/hafs/quran-recitations/elhossary_fast/035 - فاطر.mp3
        data/hafs/quran-recitations/elhossary_fast/078 - النبأ.mp3
        data/hafs/quran-recitations/elhossary_fast/001 - الفاتحة.mp3
        data/hafs/quran-recitations/053_مشاري.mp3
        data/hafs/quran-recitations/114.mp3
        data/hafs/quran-recitations/054_محمود_عبدالحكم.mp3
        data/001.mp3
    """

    def recursive_search(path: Path, files_list: list[Path]) -> None:
        if not path.exists():
            return
        if path.is_file():
            if is_audiofile(path):
                if condition_callback is None:
                    files_list.append(path)
                elif condition_callback(path):
                    files_list.append(path)
                elif delete_audiofile_on_false_cond:
                    path.unlink()

            return

        for curr_path in path.iterdir():
            if curr_path.is_file():
                if is_audiofile(curr_path):
                    if condition_callback is None:
                        files_list.append(curr_path)
                    elif condition_callback(curr_path):
                        files_list.append(curr_path)
                    elif delete_audiofile_on_false_cond:
                        curr_path.unlink()
            elif curr_path.is_dir():
                recursive_search(curr_path, files_list)

    files_list = []
    path = Path(path)
    recursive_search(path, files_list)

    return files_list


def is_audiofile(path: str | Path) -> bool:
    if get_audiofile_info(path) is not None:
        return True
    return False


@dataclass
class AudioFileInfo:
    sample_rate: int
    duration_seconds: float


def get_audiofile_info(audiofile_path: str | Path) -> AudioFileInfo:
    """Reads the file metadata and return its information
    Returns:
        (AudioFileInfo): if the audiofile is not valid return None
    """
    audio = File(audiofile_path)
    if audio is None:
        return None
    return AudioFileInfo(
        sample_rate=audio.info.sample_rate,
        duration_seconds=audio.info.length)


# @exception_catcher
def unzip_files(
    zipfile_path: str | Path,
    files: list,
    extract_path: Path
):
    """unzip part of files (files) in extract_path
    """
    # open the zip file
    with ZipFile(zipfile_path, 'r') as handle:
        # unzip multiple files
        for file in files:
            # unzip the file
            handle.extract(file, extract_path)
            # report progress
            # print(f'.unzipped {file}')


def extract_zipfile(
    zipfile_path: str | Path,
    extract_path: str | Path,
    num_workers: int = 8,
):
    # WARN: we ingore empty files in zip file
    """Extract zipfiles using multiple processes eachone is working on group of files
    source: https://superfastpython.com/multithreaded-unzip-files/
    Args:
        zipfile_path (str | Path): path to zipfile
        extract_path (str | Path): path to extract zipfile
        num_worker (int): number of worker to process zipfile in parallel

    """
    extract_path = Path(extract_path)
    # open the zip file
    files = []
    with ZipFile(zipfile_path, 'r') as handle:
        # list of all files to unzip
        # files = handle.namelist()
        for zip_info in handle.infolist():
            if not zip_info.is_dir():
                files.append(zip_info.filename)

    # creating directory structure of the unziped file
    dirs_to_make = set()
    for file in files:
        dirs_to_make.add(extract_path / Path(file).parent)
    for dir in dirs_to_make:
        os.makedirs(dir, exist_ok=True)

    # determine chunksize
    chunksize = max(len(files) // num_workers, 1)
    # start the thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        # split the copy operations into chunks
        for i in range(0, len(files), chunksize):
            # select a chunk of filenames
            selected_files = files[i:(i + chunksize)]
            # submit the batch copy task
            feature = exe.submit(unzip_files,
                                 zipfile_path, selected_files, extract_path)

    # executing this to account for errors
    feature.result()


class DownloadError(Exception):
    ...


def deduce_filename(url, verbose=False) -> str:
    """extracts filename from url
    * if the url in reachable:
        1. take the last redirected url
        2. extract file name from the last url
        3. extract extention using `filetype` package
    * if there is not internet or the url in not reachable:
        - it returns the filename the url
    """

    try:
        filename = None
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        # Read the first 2 KB of the file
        first_bytes = response.raw.read(2048)

        # Check for Content-Disposition header
        if 'content-disposition' in response.headers:
            filename = get_filename_from_header(
                response.headers['content-disposition'])
        if filename is None:
            # get filename from the redirected url
            filename = get_filename_from_url(response.url)
            if verbose:
                print(f'Faild to read header {response.url}, {filename}')
        else:
            if verbose:
                print(
                    f'Success in reading Header, header filename: {filename}')

        try:
            # trying to guess extention form first_bytes
            segs = filename.split('.')
            # old_ext = segs[-1]
            name = ''.join(segs[:-1]) if len(segs) > 1 else filename
            ext = filetype.guess_extension(first_bytes)
            if ext:
                if verbose:
                    print('Success in reading mime type')
                return name + '.' + ext
            else:
                return filename
        except Exception as e:
            if verbose:
                print(f'Error {e} while extracting file exctention for {url}')
            return filename

    except Exception as e:
        print(f'Error {e} connecting to {url}')
        return get_filename_from_url(url)


def get_filename_from_header(content_disposition) -> str | None:
    """ Gets the filename from content-disposition in GET request header
    Args:
        contnet_disposition: The respose.headers['content-disposition']

    Return:
        str | None: the filename in the header if found else `None`
    """
    # paterns to get the filename from http header
    # The priority is for the Arabic name i.e paatterns[0]
    # then ofr the English name i.e: patterns[1]
    patterns = [r'filename\*=utf-8\'\'(.*)$', r'filename="?([^$]+)"?$']
    parts = content_disposition.split(';')
    for pattern in patterns:
        for part in parts:
            match = re.search(pattern, part)
            if match:
                filename = match.group(1)
                if filename.endswith('"'):
                    filename = filename[:-1]
                return urllib.parse.unquote(filename)
    return None


def get_filename_from_url(url_link) -> str:
    """Extract filename from URL"""
    segments = urlparse(url_link).path.split('/')
    for name in segments[::-1]:
        if name:
            return urllib.parse.unquote(name)


def download_file_fast(
    url: str,
    out_path: str | Path,
    extract_zip=True,
    hash_download=False,
    num_download_segments=10,
    num_unzip_workers=12,
    remove_zipfile=True,
    redownload=False,
    show_progress=True,
    max_retries: int = 0,
) -> Path:
    """Downloads a file and extract if if it is zipfile
    Args:
        out_path (str | Path): the path to the Download (Directory)
        extract_zip (bool): if true extract a zip file to `out_path`
        remove_zipfile (bool): remove zipfile after downloading it
        redownload (bool): redownload the file if it exists
        hash_download (bool): if True the file name will be the hash(url).
            if False it will deduce the file name from the url like "001.mp3"
        max_retries (int): the number of times to try download if an error occured
            default is 0
    """
    out_path = Path(out_path)
    assert not out_path.is_file(), (
        'Download Path `out_path` has to be a directory not a file')
    os.makedirs(out_path, exist_ok=True)

    filename = deduce_filename(url)
    if hash_download:
        splits = filename.split('.')
        if len(splits) == 1:
            raise ValueError(f'The file has to extention for url: {url}')
        ext = splits[-1]
        filename = sha256(url.encode()).hexdigest() + f'.{ext}'

    out_path /= filename
    if out_path.exists() and not redownload:
        return out_path

    dl = Pypdl()
    out = dl.start(
        url,
        file_path=out_path,
        segments=num_download_segments,
        display=show_progress,
        retries=max_retries,
    )
    if out is None:
        raise DownloadError(f'Error while downloading or url: {url}')
    out_path = Path(out.path)

    if extract_zip and is_zipfile(out_path):
        zipfile_path = out_path.rename(
            out_path.parent / f'{out_path.name}.download')
        extract_zipfile(zipfile_path=zipfile_path,
                        extract_path=out_path, num_workers=num_unzip_workers)

        # remove zipfile
        if remove_zipfile:
            zipfile_path.unlink()

    return out_path
