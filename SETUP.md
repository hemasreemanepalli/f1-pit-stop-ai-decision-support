# SETUP.md — running this project on your own computer

These instructions assume no programming experience. They were written
for a normal Windows / Mac / Linux computer with nothing special
installed.

## 1. Install Python

You need Python 3.10 or newer.

- **Windows / Mac**: download from https://www.python.org/downloads/
  and run the installer. On Windows, tick **"Add Python to PATH"**
  during install.
- **Mac (alternative)**: if you have [Homebrew](https://brew.sh),
  `brew install python`.
- **Linux**: Python 3 is usually already installed
  (`python3 --version` to check).

To check it worked, open a terminal (Mac: Terminal app; Windows:
Command Prompt or PowerShell) and type:

```bash
python --version
```

If that says "command not found", try `python3 --version` instead —
use whichever one works for the rest of these steps.

## 2. Get the project folder

Copy the whole `f1-pit-stop-ai` folder onto your computer (e.g. onto
your Desktop). Open a terminal and move into it:

```bash
cd Desktop/f1-pit-stop-ai
```

(Adjust the path if you put it somewhere else.)

## 3. (Recommended) create a virtual environment

This keeps this project's Python packages separate from anything else
on your computer.

```bash
python -m venv venv
```

Then activate it:

- **Mac/Linux**: `source venv/bin/activate`
- **Windows**: `venv\Scripts\activate`

You should see `(venv)` appear at the start of your terminal prompt.
Do this every time you open a new terminal to work on this project.

## 4. Install the required packages

```bash
pip install -r requirements.txt
```

This downloads everything the project needs (takes a few minutes the
first time).

## 5. Run the pipeline

The simplest, guaranteed-to-work option (uses only the provided CSV):

```bash
python run_pipeline.py --offline
```

This takes about 2 minutes and prints progress for each of the 8
stages. When it finishes, look in the `reports/` folder for the
results, and `models/` for the trained model files.

### Optional: also try fetching live F1 data (Jolpica API)

```bash
python run_pipeline.py --download
```

This additionally tries to download and cache real per-lap data from
the Jolpica-F1 API before running the rest of the pipeline. It needs a
normal internet connection. **Important honesty note:** this code was
written and tested against realistic mocked data, but the developer's
own environment could not reach this API to verify it against a real
response (see README.md §2/§9). The first time you run `--download`:

1. Check the terminal output — if it says "Jolpica download failed",
   the pipeline automatically falls back to CSV-only mode and still
   completes normally.
2. If it succeeds, look inside `data/raw/jolpica_cache/` — you should
   see files like `laps_2023_1.json`. Open one in a text editor and
   sanity-check that it looks like real F1 lap data (lap numbers,
   driver names, lap times).
3. Re-run `python run_pipeline.py` (without `--offline` or
   `--download`) to rebuild everything using that cached data.

If something looks wrong with the downloaded data, please treat the
CSV-only results (`--offline`) as the trustworthy ones — those are
what this project's README numbers are based on.

## 6. View the dashboard

```bash
streamlit run dashboard/app.py
```

This opens a web page (usually at `http://localhost:8501`) in your
browser automatically. Use the sidebar to pick a season, race, driver,
and lap number.

If your browser doesn't open automatically, copy the "Local URL" shown
in the terminal into your browser manually.

To stop the dashboard, go back to the terminal and press `Ctrl+C`.

## 7. Run the automated tests (optional, checks everything still works)

```bash
pytest
```

You should see something like `33 passed, 1 skipped`. If anything
shows `FAILED`, something in your environment differs from the one
this was built in — check the error message, or re-run
`pip install -r requirements.txt` in case a package failed to install.

## Troubleshooting

- **"python: command not found"** → try `python3` instead of `python`
  everywhere in these instructions.
- **"pip: command not found"** → try `pip3` instead of `pip`, or
  `python -m pip install -r requirements.txt`.
- **Streamlit doesn't open a browser** → manually visit
  `http://localhost:8501`.
- **Something about `pyarrow` or `parquet` fails to install** → this is
  usually a Python-version mismatch; make sure you're on Python 3.10+.
- **Everything is very slow** → the first `pip install` step is the
  slow one (downloading packages); the pipeline itself takes about 2
  minutes once packages are installed.
