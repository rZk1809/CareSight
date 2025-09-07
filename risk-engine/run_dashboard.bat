@echo off
echo 🚀 Starting CareSight Risk Engine Dashboard...
echo 🌐 Dashboard will be available at: http://localhost:8501
echo 💡 Press Ctrl+C to stop the dashboard
echo.
echo ============================================================
echo.

REM Set Python path
set PYTHONPATH=%CD%\src

REM Run Streamlit dashboard
python -m streamlit run src\dashboards\streamlit_app.py --server.address=localhost --server.port=8501

pause
