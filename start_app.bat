@echo off
if not exist env (
    echo Creating virtual environment...
    py -m venv env
)
call env\Scripts\activate.bat
echo Installing requirements...
pip install -r requirements.txt
echo Running application...
py src\yolo11.py
