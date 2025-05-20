import subprocess
import sys
import importlib.util

def install_library(library_name):
    """تثبيت مكتبة إذا لم تكن مثبتة بالفعل."""
    if importlib.util.find_spec(library_name) is None:
        print(f"{library_name} غير مثبتة. جارٍ التثبيت...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    else:
        print(f"{library_name} مثبتة بالفعل.")

# قائمة بالمكتبات المطلوبة
required_libraries = [
    "scikit-learn",
    "streamlit",
    "pandas",
    "yfinance",
    "TA-Lib",
    "numpy",
    "requests",
    "textblob",
    "statsmodels",
    "scikit-learn",
    "tensorflow",
    "transformers",
    "stable-baselines3",
    "gym",
    "pycuda",
    "bayesian-optimization",
    "deap",
    "shap",
    "stellargraph",
    "keras-tuner",
    "tensorflow-model-optimization",
    "plotly",
    "dash",
    "alpaca-trade-api",
    "smtplib"
]

# تثبيت المكتبات المطلوبة تلقائيًا
for library in required_libraries:
    install_library(library)

print("تم تثبيت جميع المكتبات المطلوبة.")