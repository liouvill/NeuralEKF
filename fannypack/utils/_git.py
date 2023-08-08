import os
import subprocess


def get_git_commit_hash(path: str = "./") -> str:
    """Returns the current Git commit hash.

    Args:
        path (str, optional): Path to check. Defaults to './'.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(path))
        .decode("ascii")
        .strip()
    )
