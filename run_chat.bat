@echo off
:: Ensure the script runs in its own directory so paths resolve correctly when double-clicked
cd /d "%~dp0"

title Orbyte AI - Legal RAG System CLI
echo =======================================================================
echo          Starting Orbyte AI Multi-Agent Legal RAG System CLI
echo =======================================================================
echo.

if not exist .venv goto :err_venv

:: Run the query_db.py script to start the interactive chat loop
.venv\Scripts\python.exe rag\query_db.py %*

echo.
echo =======================================================================
echo          Orbyte AI Session Terminated.
echo =======================================================================
pause
exit /b 0

:err_venv
echo [ERROR] Virtual environment (.venv) not found.
echo Please make sure you have run the setup and created the virtual environment.
pause
exit /b 1
