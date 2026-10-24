@echo off
cd /d D:\work\live_predict
call venv\Scripts\activate.bat

echo ============================================================
echo  Live Predict Batch Runner  (every 15 minutes)
echo  Voyages: 31  ^|  DB: predictions\predictions.db
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

:loop
echo [%DATE% %TIME%] --- Running batch predictions...
python batch_predict.py
echo [%DATE% %TIME%] --- Done. Waiting 15 minutes (Ctrl+C to abort)...
timeout /t 900 /nobreak
goto loop
