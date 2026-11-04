"""Rebuild and execute Lab1.ipynb using this Python interpreter as the kernel."""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent


def main():
    runpy.run_path(str(ROOT / "build_notebook.py"), run_name="__main__")
    notebook = nbformat.read(ROOT / "Lab1.ipynb", as_version=4)
    # A temporary kernel spec avoids registering anything in the user's account.
    with tempfile.TemporaryDirectory(prefix="ml-lab-1-kernel-") as temporary:
        kernel_directory = Path(temporary) / "kernels" / "ml-lab-1-run"
        kernel_directory.mkdir(parents=True)
        (kernel_directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ML lab 1 execution", "language": "python",
        }), encoding="utf-8")
        previous_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary + (os.pathsep + previous_path if previous_path else "")
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        try:
            client = NotebookClient(notebook, timeout=1200, kernel_name="ml-lab-1-run",
                                    resources={"metadata": {"path": str(ROOT)}})

            def report(cell_index, **kwargs):
                print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

            client.on_cell_start = report
            client.execute()
        finally:
            nbformat.write(notebook, ROOT / "Lab1.ipynb")
            if previous_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_path
    print("Executed Lab1.ipynb; predictions saved to submission.csv.")
    print((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
