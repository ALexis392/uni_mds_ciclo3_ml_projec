"""
Main Application
Ejecuta la API FastAPI + interfaz web HTML
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
import webbrowser
import time
import threading

def open_browser():
    """Abre el navegador automáticamente después de 2 segundos"""
    time.sleep(2)
    webbrowser.open('http://localhost:8888')  # CAMBIO: 8000 → 8888

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 INICIANDO FRAUD DETECTION API")
    print("="*70)
    print("\n📍 URL: http://localhost:8888")  # CAMBIO: 8000 → 8888
    print("📚 Documentación: http://localhost:8888/docs")  # CAMBIO: 8000 → 8888
    print("📊 ReDoc: http://localhost:8888/redoc")  # CAMBIO: 8000 → 8888
    print("\nAbriendo navegador automáticamente...\n")
    print("="*70 + "\n")
    
    # Abrir navegador en background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Iniciar API
    os.chdir(project_root)
    uvicorn.run(
        "src.serving:app",
        host="0.0.0.0",
        port=8888,  # CAMBIO: 8000 → 8888
        reload=False,
        log_level="info"
    )