@echo off
echo ========================================
echo   Sakura Companion - 养老陪伴AI
echo ========================================
echo.
echo Starting ArrowCanaria inference (:8014)...
start "ArrowCanaria" cmd /c "python arrowcanaria_server\server.py"
echo Starting Companion service (:8015)...
python -m companion.main
pause
