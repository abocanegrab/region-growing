#!/usr/bin/env python3
"""
Test Sentinel Hub connection with real API call.
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.sentinel_hub_service import SentinelHubService
from app.utils import get_logger

logger = get_logger(__name__)

def main():
    """Test Sentinel Hub connection"""
    print("=" * 60)
    print("Probando conexion con Sentinel Hub")
    print("=" * 60)

    try:
        # Initialize service
        print("\n[1/3] Inicializando servicio...")
        service = SentinelHubService()
        print("   [OK] Servicio inicializado")

        # Test connection
        print("\n[2/3] Probando conexion con API...")
        result = service.test_connection()

        if result['status'] == 'success':
            print(f"   [OK] Conexion exitosa!")
            print(f"   Data shape: {result.get('data_shape', 'N/A')}")
            print(f"   Message: {result['message']}")
            print("\n" + "=" * 60)
            print("[SUCCESS] Sentinel Hub configurado correctamente!")
            print("=" * 60)
            return 0
        else:
            print(f"   [ERROR] Conexion fallida")
            print(f"   Message: {result['message']}")
            print("\n" + "=" * 60)
            print("[ERROR] Error al conectar con Sentinel Hub")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n[ERROR] Excepcion durante la prueba:")
        print(f"   {str(e)}")
        logger.error("Error testing Sentinel Hub connection", exc_info=True)
        print("\n" + "=" * 60)
        print("[ERROR] Prueba fallida")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
