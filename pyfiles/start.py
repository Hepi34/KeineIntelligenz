# Hepi34, 2026-03-06

import os
import sys
import subprocess

OS = sys.platform

if os.path.exists(".venv"):
    if OS == "win32":
        try:
            os.system(".venv\\Scripts\\activate")
            print("Virtual environment activated.")
        except Exception as e:
            print(f"Error activating virtual environment: {e}")
    else:
        try:
            os.system("source .venv/bin/activate")
            print("Virtual environment activated.")
        except Exception as e:
            print(f"Error activating virtual environment: {e}")

else:
    if OS == "win32":
        try:
            os.system("python -m venv .venv")
            os.system(".venv\\Scripts\\activate")
            print("Virtual environment created and activated.")
        except Exception as e:
            print(f"Error setting up virtual environment: {e}")
    else:
        try:
            os.system("python3 -m venv .venv")
            os.system("source .venv/bin/activate")
            print("Virtual environment created and activated.")
        except Exception as e:
            print(f"Error setting up virtual environment: {e}")

if OS == "win32":
    with open("requirements.txt", "r") as f:
        for line in f:
            if os.path.exists(".venv\\Lib\\site-packages\\" + line.strip()):
                print(f"Package {line.strip()} is already installed.")
            else:
                os.system(f".venv\\Scripts\\pip install {line.strip()}")
else:
    with open("requirements.txt", "r") as f:
        pyver = subprocess.check_output("ls .venv/lib/ | head -n 1", shell=True, text=True).strip()
        for line in f:
            if os.path.exists(f".venv/lib/{pyver}/site-packages/{line.strip()}"):
                print(f"Package {line.strip()} is already installed.")
            else:
                os.system(f".venv/bin/pip3 install {line.strip()}")


if OS == "win32":
    try:
        os.system(".venv\\Scripts\\python pyfiles/gui.py")
    except Exception as e:
        print(f"Error running gui.py: {e}")
else:
    try:
        os.system(".venv/bin/python3 pyfiles/gui.py")
    except Exception as e:
        print(f"Error running gui.py: {e}")    