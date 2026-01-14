from pathlib import Path


def validate_file_exists(path: str | Path) -> Path:
    """Validates that a file exists and returns the Path object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path


def validate_dir_exists(path: str | Path) -> Path:
    """Validates that a directory exists and returns the Path object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return path


def ensure_dir(path: str | Path) -> Path:
    """Ensures a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path, extensions: set | None = None) -> list[Path]:
    """Lists files in a directory, optionally filtering by extension."""
    directory = validate_dir_exists(directory)
    files = [f for f in directory.rglob("*") if f.is_file()]
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return sorted(files)
