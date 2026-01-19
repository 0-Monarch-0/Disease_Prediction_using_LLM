@echo off
:: ============================================================
:: Start the Disease & Case Prediction FastAPI server
:: ============================================================

echo --------------------------------------------------------
echo Activating environment and starting FastAPI server...
echo --------------------------------------------------------

:: Activate the virtual environment (use 'call' to ensure persistence)
call D:\DLProject\.venv\Scripts\activate

:: Change directory to the API folder (use /d to switch drives if needed)
cd /d D:\disease-predictor\api

:: Run FastAPI using the same Python interpreter from the venv
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo --------------------------------------------------------
echo Server stopped or closed.
pause
