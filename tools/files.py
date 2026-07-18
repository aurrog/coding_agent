from pathlib import Path
from config import *


def read_file(path):
    with open(path, 'r') as f:
        return f.read()
    


def list_files(root,max_depth=3,depth=0):
    root = Path(root)
    result=[]

    if depth>max_depth:
        return result
    
    for item in root.iterdir():
        if item.name in IGNORE_DIRS:
            continue

        result.append({
            'path': str(item),
            'type': 'directory' if item.is_dir() else 'file'
        })

        if item.is_dir():
            result.extend(
                list_files(
                    item,
                    max_depth,
                    depth+1
                )
            )
    return result


def edit_file(path, content):
    file=Path(path)
    file.write_text(
        content,
        encoding='utf-8'
    )
    