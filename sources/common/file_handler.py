from email.mime import base
from pathlib import Path
import json
from os import path


def open_file_from_fullpath(file_path, mode):
    filepath = Path(file_path)
    file = open(filepath, mode)
    return file



def open_file_from_path(path, filename, mode):
    path = Path(path)
    filepath = path / filename
    file = open(filepath, mode)
    return file


def ask_file_path():
    raw_path = input("Enter the path of the file ")
    return raw_path


def get_value_from_json(file, key):
    value = json.loads(file.read())
    file.seek(0)
    return value[key]


def get_file_name(file):
    return path.splitext(path.basename(file.name))[0]



def get_file_extension(file):
    return path.splitext(file.name)[1]


def get_file_path(file):
    return path.dirname(file.name)


def merge_path_filename(path, base_name, suffix, extension):
    return str(Path(path + "/" + base_name + suffix + extension))


def generate_new_file_path(file, suffix):
    path = get_file_path(file)
    base_name = get_file_name(file)
    extension = get_file_extension(file)
    return merge_path_filename(path=path, base_name=base_name, suffix=suffix, extension=extension)