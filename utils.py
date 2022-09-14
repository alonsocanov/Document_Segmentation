import glob
import os


def filesInDir(path: str, ext: str = '') -> list:
    if ext:
        if not ext.startswith('.'):
            ext = '.'+ext
        ext = '*'+ext
        path = os.path.join(path, ext)
    return glob.glob(path, recursive=True)


def getParentDir():
    return os.path.abspath(os.path.dirname(__file__))


def joinPath(*args):
    return os.path.join(*args)
