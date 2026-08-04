from pathlib import Path

# Absolute project file path, for example: d:/temp/proj.xlsx
PROJECT_FILE_PATH: str | None = None

# Absolute project directory path, for example: d:/temp
PROJECT_DIR_PATH: str | None = None


def set_project_file_path(file_path: str | None) -> None:
    """Update the global project file path and parent directory."""
    global PROJECT_FILE_PATH, PROJECT_DIR_PATH

    if not file_path:
        PROJECT_FILE_PATH = None
        PROJECT_DIR_PATH = None
        return

    resolved = Path(file_path).expanduser().resolve()
    PROJECT_FILE_PATH = str(resolved)
    PROJECT_DIR_PATH = str(resolved.parent)


def get_project_file_path() -> Path | None:
    if PROJECT_FILE_PATH is None:
        return None
    return Path(PROJECT_FILE_PATH)


def get_project_directory() -> Path | None:
    if PROJECT_DIR_PATH is None:
        return None
    return Path(PROJECT_DIR_PATH)
