"""Validate bilingual lesson files and optionally execute fresh Jupyter kernels."""
import argparse
import base64
import copy
import json
import platform
import re
import tempfile
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
PILOT_CHECK = '''
assert shown_state["distance_m"] == 6
assert experiment_state["distance_m"] == 6
assert exercise_passed is False  # The authored exercise is intentionally unfinished.
assert solution_distance_m == 12
assert check_distance_result(3, 4, 12) is True
assert check_distance_result(3, 4, 11) is False
assert check_distance_result(0, 5, 0) is True
assert check_distance_result(2, 0, 0) is True
assert check_distance_result(-1, 2, -2) is False
assert check_distance_result(2, -1, -2) is False
assert check_distance_result(3, 4, "12") is False
assert check_distance_result(3, 4, float("nan")) is False
assert check_distance_result(3, 4, True) is False
exercise_distance_m = exercise_speed_mps * exercise_time_s
assert check_distance_result(exercise_speed_mps, exercise_time_s, exercise_distance_m)
zero_state = show_distance_flow(0, 5)
assert zero_state["distance_m"] == 0
print("Authoring verification: PASS")
'''


def validate_pair(korean, english):
    for notebook in (korean, english):
        nbformat.validate(notebook)
        ids = [cell.id for cell in notebook.cells]
        assert len(ids) == len(set(ids)), "Duplicate cell IDs"
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None and not cell.outputs, "Clear saved outputs"
    ko = [(cell.id, cell.source) for cell in korean.cells if cell.cell_type == "code"]
    en = [(cell.id, cell.source) for cell in english.cells if cell.cell_type == "code"]
    assert ko == en, "Bilingual code cells differ"
    question_ids = lambda book: sorted(set(re.findall(r"PP\d+-(?:Q|E)\d+", "\n".join(cell.source for cell in book.cells))))
    assert question_ids(korean) == question_ids(english), "Bilingual question IDs differ"
    for field in ("course_id", "lesson_id"):
        assert korean.metadata.suanlab[field] == english.metadata.suanlab[field]
    assert korean.metadata.suanlab.locale == "ko" and english.metadata.suanlab.locale == "en"


def execute(path, original, output_directory):
    from nbclient import NotebookClient

    notebook = copy.deepcopy(original)
    pilot = notebook.metadata.suanlab.lesson_id == "pp-01-first-program"
    if pilot:
        notebook.cells.append(nbformat.v4.new_code_cell(PILOT_CHECK, id="authoring-verification"))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="suanlab-notebook-") as directory:
        NotebookClient(notebook, timeout=120, kernel_name="python3",
                       resources={"metadata": {"path": directory}}).execute()
    images = [cell_output.data["image/png"] for cell in notebook.cells
              if cell.cell_type == "code" for cell_output in cell.outputs
              if cell_output.output_type in ("display_data", "execute_result")
              and "image/png" in cell_output.data]
    if pilot:
        assert len(images) >= 3, "Expected baseline, experiment and boundary-case figures"
        exercise_cell = next(cell for cell in notebook.cells if cell.id == "exercise")
        text = "".join(item.get("text", "") for item in exercise_cell.outputs)
        assert "TRY_AGAIN" in text, "Unfinished starter must not pass"
    nbformat.write(notebook, output_directory / path.name)
    if images:
        (output_directory / (path.stem + ".png")).write_bytes(base64.b64decode(images[0]))
    return {"path": str(path.relative_to(ROOT)), "code_cells": sum(c.cell_type == "code" for c in original.cells),
            "png_outputs": len(images), "seconds": round(time.perf_counter() - started, 2),
            "local_jupyter": "passed", "hosted_colab": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute all paired notebooks in fresh local kernels")
    args = parser.parse_args()
    directory = ROOT / "notebooks/python-programming"
    korean_paths = sorted(directory.glob("*.ko.ipynb"))
    assert korean_paths, "No lessons found"
    rows = []
    output_directory = ROOT / ".runtime/learning-check"
    if args.execute:
        output_directory.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    for ko_path in korean_paths:
        en_path = ko_path.with_name(ko_path.name.replace(".ko.ipynb", ".en.ipynb"))
        expected_paths.update((ko_path, en_path))
        korean = nbformat.read(ko_path, as_version=4)
        english = nbformat.read(en_path, as_version=4)
        validate_pair(korean, english)
        for path, notebook in ((ko_path, korean), (en_path, english)):
            if args.execute:
                result = execute(path, notebook, output_directory)
                rows.append(result)
                print(json.dumps(result), flush=True)
    assert set(directory.glob("*.ipynb")) == expected_paths, "Unpaired or unexpected notebook"
    if args.execute:
        import ipykernel
        import matplotlib
        import nbclient
        report = {"environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                  "nbformat": nbformat.__version__, "nbclient": nbclient.__version__,
                                  "ipykernel": ipykernel.__version__}, "notebooks": rows}
        (output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(expected_paths)} notebook files; bilingual code cells match.")


if __name__ == "__main__":
    main()
