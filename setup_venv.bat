@echo off
cd /d c:\Users\Kingpin\Downloads\Projects\Vilvom_Application\TeaLeaf-YOLOv11
echo Removing old virtual environment...
rmdir /s /q env
echo Creating new virtual environment...
python -m venv env
echo Installing requirements...
call env\Scripts\activate.bat
pip install -r requirements.txt
echo Virtual environment setup complete!
pause
