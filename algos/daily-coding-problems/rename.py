import pathlib
import sys
import shutil

dry_run = "-d" in sys.argv

dir = pathlib.Path("./solutions")
files = dir.glob("*")
for path in files:
    if len(path.stem) < 4:
        new_name = ("0" * 4 + path.stem)[-4:] + path.suffix
        print(f"{path.name} -> {new_name}")
        if not dry_run:
            shutil.move(f"./solutions/{path.name}", f"./solutions/{new_name}")
