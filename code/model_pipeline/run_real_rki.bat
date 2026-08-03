@echo off
setlocal
set HERE=%~dp0
for %%I in ("%HERE%\..\..") do set REPO=%%~fI
python "%REPO%\scripts\reproduce_data.py"
if errorlevel 1 exit /b 1
python "%HERE%\run_all.py" --data-root "%REPO%" --mode all --output "%REPO%\outputs\recomputed_full"
if errorlevel 1 exit /b 1
endlocal
