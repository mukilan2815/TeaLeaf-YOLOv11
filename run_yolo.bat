@echo off
cd /d c:\Users\Kingpin\Downloads\Projects\Vilvom_Application\TeaLeaf-YOLOv11
echo Activating virtual environment...
call env\Scripts\activate.bat
echo Running YOLO script...
python.exe src\yolo11.py
pause
