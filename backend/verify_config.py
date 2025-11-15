#!/usr/bin/env python3
"""
Quick script to verify Sentinel Hub credentials are loaded correctly.
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Settings, ENV_FILE

def main():
    """Verify configuration loading"""
    print("=" * 60)
    print("Verificando configuracion de Sentinel Hub")
    print("=" * 60)

    # Check if .env file exists
    print(f"\n[FILE] Archivo .env: {ENV_FILE}")
    print(f"   Existe: {'[OK] Si' if ENV_FILE.exists() else '[ERROR] No'}")

    # Load settings
    settings = Settings()

    # Check credentials
    print(f"\n[CREDENTIALS] Credenciales:")
    print(f"   Client ID: {settings.sentinelhub_client_id[:20]}..." if settings.sentinelhub_client_id else "   Client ID: [ERROR] No configurado")
    print(f"   Client Secret: {settings.sentinelhub_client_secret[:10]}..." if settings.sentinelhub_client_secret else "   Client Secret: [ERROR] No configurado")

    # Validate
    is_valid = settings.validate_credentials()
    print(f"\n[STATUS] Estado: {'Configuracion valida' if is_valid else '[ERROR] Configuracion incompleta'}")

    # Test backward compatibility
    print(f"\n[COMPATIBILITY] Compatibilidad hacia atras:")
    print(f"   sentinel_hub_client_id (property): {settings.sentinel_hub_client_id[:20]}..." if settings.sentinel_hub_client_id else "   [ERROR] Error")
    print(f"   sentinel_hub_client_secret (property): {settings.sentinel_hub_client_secret[:10]}..." if settings.sentinel_hub_client_secret else "   [ERROR] Error")

    print("\n" + "=" * 60)

    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())
