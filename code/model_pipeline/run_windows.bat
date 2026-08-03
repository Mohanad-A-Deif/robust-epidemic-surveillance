@echo off
setlocal
set HERE=%~dp0
for %%I in ("%HERE%\..\..") do set REPO=%%~fI
python -m pip install -r "%REPO%\requirements.txt"
if errorlevel 1 exit /b 1
python "%HERE%\run_all.py" --data-root "%REPO%" --mode all --quick --output "%REPO%\outputs\quick_real_data"
if errorlevel 1 exit /b 1
endlocal
