@echo off
python -m pip install -r requirements.txt
python scripts\run_tests.py
python scripts\validate_repository.py
pause
