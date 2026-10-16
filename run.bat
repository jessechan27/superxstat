@echo off
rem Super Xtat - double-click to start; the browser opens http://localhost:8501
rem Listens on localhost only: the Code page runs Python on this computer.
cd /d "%~dp0"
python -m streamlit run app.py --server.address localhost
pause
