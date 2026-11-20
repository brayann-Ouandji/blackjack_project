@echo off
cd /d "%~dp0"
python app.py --source 1 --weights vision/weights/best.pt --q runs/run2/q.pkl
pause