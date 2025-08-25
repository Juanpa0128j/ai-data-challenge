@echo off
echo 🚀 Activando entorno Conda para AI Data Challenge...
call conda activate ai-data-challenge

echo 📊 Iniciando API Flask en el puerto 5000...
echo 🌐 API endpoints disponibles:
echo    - Health check: http://localhost:5000/api/health
echo    - Prediccion: http://localhost:5000/api/predict
echo    - Info modelo: http://localhost:5000/api/model-info  
echo    - Ejemplos demo: http://localhost:5000/api/demo-examples
echo    - Estadisticas: http://localhost:5000/api/statistics
echo.
echo ⚠️  Presiona Ctrl+C para detener el servidor
echo.

cd /d "d:\Coding\Projects\ai-data-challenge\api"
python flask_api.py

pause
