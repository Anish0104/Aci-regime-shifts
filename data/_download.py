"""Raw download cache shared by the data loaders."""
from pathlib import Path
from urllib.request import urlopen
import shutil

RAW_DIR = Path(__file__).resolve().parent / "raw"


def download(url, filename):
    """Download once, publishing the cache file only after success."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / filename
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target
