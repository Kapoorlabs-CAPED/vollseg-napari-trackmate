import subprocess
import re

def update_version_files():
    # Get the latest Git tag
    tag = (
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .decode()
        .strip()
    )
    if "v" in tag:
        tag = tag.lstrip("v")

    # Update _version.py
    with open("src/vollseg_napari_trackmate/_version.py", "w") as version_file:
        version_file.write(f'__version__ = version = "{tag}"\n')
        version_file.write(
            f'__version_tuple__ = version_tuple = {tuple(map(int, tag.split(".")))}\n'
        )

    # Update setup.cfg
    with open("setup.cfg", "r") as setup_cfg_file:
        setup_cfg = setup_cfg_file.read()

    setup_cfg = re.sub(
        r"version\s*=\s*[\'\"]([^\'\"]*)[\'\"]",
        f'version = "{tag}"',
        setup_cfg,
        count=1,
    )

    with open("setup.cfg", "w") as setup_cfg_file:
        setup_cfg_file.write(setup_cfg)

if __name__ == "__main__":
    update_version_files()
