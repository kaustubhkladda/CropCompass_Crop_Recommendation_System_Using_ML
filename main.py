"""
CropCompass - launcher (set this as the "Main file path" on Streamlit Cloud).

It does two jobs:
  1. Checks that every required package is installed and, if not, shows a clear
     message on the page instead of a redacted crash.
  2. Runs app.py, which contains the actual application.
"""

import importlib
import runpy
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
APP_FILE = HERE / "app.py"
DATA_FILE = HERE / "Crop_recommendation.csv"

# import name -> name to put in requirements.txt
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
}

missing = []
for module, pip_name in REQUIRED.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(pip_name)

if missing:
    st.set_page_config(page_title="CropCompass - setup needed", page_icon="🧭")
    st.title("🧭 CropCompass")
    st.error("Some required packages are not installed on this server.")
    st.write("Missing packages: " + ", ".join(f"`{m}`" for m in missing))
    st.write(
        "Make sure a file named **requirements.txt** sits in the root of your GitHub repo "
        "(next to main.py) and contains:"
    )
    st.code("\n".join(["streamlit"] + list(REQUIRED.values())), language="text")
    st.write("Then reboot the app from the Streamlit Cloud dashboard.")
    st.stop()

if not APP_FILE.exists():
    st.error("app.py was not found next to main.py. Both files must be in the same folder of the repo.")
    st.stop()

if not DATA_FILE.exists():
    st.error("Crop_recommendation.csv was not found next to main.py. Please add it to the repo root.")
    st.stop()

# Run the app. Streamlit re-executes this launcher on every interaction,
# and the cached model training inside app.py is reused between runs.
runpy.run_path(str(APP_FILE), run_name="__main__")
