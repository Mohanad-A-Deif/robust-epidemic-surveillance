@echo off
setlocal
set HERE=%~dp0
for %%I in ("%HERE%\..\..") do set REPO=%%~fI
python -m pip install -r "%REPO%\requirements.txt"
if errorlevel 1 exit /b 1
python "%REPO%\scripts\reproduce_data.py" %*
endlocal
