import pathlib
import shutil


dir = pathlib.Path('./solutions')
files = dir.glob("*")
for path in files:
    if len(path.name) < 4:
        new_name = ('0'*4 + path.name)[-4:-1]
        shutil.move(f'./solutions/{path.name}', f'./solutions/{new_name}')
