import os
import sys

OS = sys.platform

if os.path.exists(".venv"):
    if OS == "win32":
        try:
            os.system(".venv\\Scripts\\activate")
        except Exception as e:
            print(f"Error activating virtual environment: {e}")
    else:
        try:
            os.system("source .venv/bin/activate")
        except Exception as e:
            print(f"Error activating virtual environment: {e}")

else:
    if OS == "win32":
        try:
            os.system("python -m venv .venv")
            os.system(".venv\\Scripts\\activate")
        except Exception as e:
            print(f"Error setting up virtual environment: {e}")
    else:
        try:
            os.system("python3 -m venv .venv")
            os.system("source .venv/bin/activate")
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
        for line in f:
            if os.path.exists(".venv/lib/python3.10/site-packages/" + line.strip()):
                print(f"Package {line.strip()} is already installed.")
            else:
                os.system(f"source .venv/bin/pip install {line.strip()}")


if OS == "win32":
    try:
        os.system("python pyfiles/gui.py")
    except Exception as e:
        print(f"Error running gui.py: {e}")
else:
    try:
        os.system("python3 pyfiles/gui.py")
    except Exception as e:
        print(f"Error running gui.py: {e}")    