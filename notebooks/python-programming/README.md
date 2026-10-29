# Python Programming Colab lessons

This is the first Colab course being produced. The current pilot is **01 First Program
and the Flow of Values**. Lessons 02–09 are planned in the
[course design](../../docs/online-learning/python-programming.md).

| Language | Pilot notebook | Execution status |
| --- | --- | --- |
| Korean | [01-first-program.ko.ipynb](01-first-program.ko.ipynb) | See local verification below; hosted Colab pending |
| English | [01-first-program.en.ipynb](01-first-program.en.ipynb) | See local verification below; hosted Colab pending |

Download a notebook and use Colab's Upload notebook option. A GitHub-backed Colab URL
can be added after these files are published to the repository. These files do not yet
establish a live website lesson or a verified hosted Colab session.

## What to try

1. Run every cell from the top in a CPU runtime.
2. Predict expression results, then execute them.
3. Change speed or time and rerun the calculation and plotting cells below it.
4. Replace the exercise's initial answer `0` with an expression using its input variables.
5. Read the explanation and compare with the unit-labelled diagram and graph.

The model assumes constant, non-negative speed and time. Both notebooks share all code,
question IDs and numeric answers; explanations are translated. Plot labels use canonical
variable names and SI units. The plotting and feedback helpers are provided code, while
functions, lists and loops are introduced in later lessons.

The unchanged exercise deliberately prints `TRY_AGAIN`. Running the worked solution or
reference checks does not complete the learner's answer. Change the exercise expression
and run it again to get `PASS`. Feedback is for practice, not a course grade.

## Verification record

On 2026-10-26, both versions passed execution of all 11 authored code cells in fresh
local Jupyter kernels. The checks also exercised correct, incorrect, zero and invalid
inputs, the intentionally unfinished exercise, and rendered diagram outputs.
The local environment was Python 3.12.3, Matplotlib 3.11.2, nbformat 5.11.1,
nbclient 0.11.0 and ipykernel 7.4.0. Hosted Colab verification is pending.

## Local verification

From the repository root:

```bash
python3 -m venv .runtime/notebooks-venv
.runtime/notebooks-venv/bin/python -m pip install -r scripts/learning/requirements.txt
.runtime/notebooks-venv/bin/python scripts/learning/check_notebooks.py --execute
```

The checker validates the Jupyter file format, matching code cells and clean saved outputs.
With `--execute`, it starts a fresh kernel for each notebook in a temporary working directory,
checks the default calculation and feedback, then tries correct, incorrect and boundary inputs.
Executed copies, diagram previews and an environment report go to `.runtime/learning-check/`.
The authored notebooks stay unexecuted with no saved output or personal kernel state.

A local pass is distinct from a hosted Colab pass. The next publication gate is running both
versions in new Colab CPU runtimes and recording that date and environment separately.

Format and execution references: [Jupyter notebook format](https://nbformat.readthedocs.io/en/latest/format_description.html)
and [NBClient execution](https://nbclient.readthedocs.io/en/latest/client.html).
