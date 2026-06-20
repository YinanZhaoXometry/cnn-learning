import shutil
import urllib.request
import zipfile
from pathlib import Path

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_and_extract(data_dir: Path) -> Path:
    zip_path = data_dir / "tiny-imagenet-200.zip"
    root = data_dir / "tiny-imagenet-200"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(_URL, zip_path)

    if not root.exists():
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    return root


def _reorganize_val(root: Path) -> None:
    """Copy val images into per-class subdirs so ImageFolder can read them."""
    val_by_class = root / "val_by_class"
    if val_by_class.exists():
        return

    print("Preparing val_by_class for ImageFolder...")
    val_by_class.mkdir()
    ann_file = root / "val" / "val_annotations.txt"
    val_img_dir = root / "val" / "images"

    with open(ann_file) as f:
        for line in f:
            img, cls = line.split()[:2]
            dst = val_by_class / cls
            dst.mkdir(exist_ok=True)
            shutil.copy2(val_img_dir / img, dst / img)


def prepare_tiny_imagenet_data(
    source: str = "local",
    local_dir: str = "./dataset-tiny-imagenet-200",
    data_dir: str = "./data",
) -> Path:
    """Return the dataset root, downloading/extracting if necessary.

    source="local"  — use pre-extracted data at local_dir (no download).
    source="remote" — download from Stanford and extract into data_dir.
    """
    if source == "local":
        root = Path(local_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"Local dataset not found: {root}\n"
                "Extract tiny-imagenet-200 manually or use source='remote'."
            )
        print(f"Using local dataset: {root}")
    elif source == "remote":
        root = _download_and_extract(Path(data_dir))
    else:
        raise ValueError(f"source must be 'local' or 'remote', got {source!r}")

    _reorganize_val(root)
    return root
