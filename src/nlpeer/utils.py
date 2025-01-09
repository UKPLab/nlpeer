from os import walk, path


def list_files(dir_path, level=1):
    for e in walk(dir_path):
        level -= 1
        if level > 0:
            continue

        files = [path.join(path.abspath(e[0]), f) for f in e[2]]
        for f in files:
            yield f
        break


def list_dirs(dir_path, level=1):
    for e in walk(dir_path):
        level -= 1
        if level > 0:
            continue

        files = [path.join(path.abspath(e[0]), f) for f in e[1]]
        for f in files:
            yield f
        break


