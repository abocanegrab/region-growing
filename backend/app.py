"""
FastAPI application entry point
"""
from config.config import Settings
import uvicorn

settings = Settings()

if __name__ == '__main__':
    # Validate configuration before starting
    try:
        if not settings.validate_credentials():
            print("[WARNING] Sentinel Hub credentials not configured")
            print("Tip: Configure SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET in .env")
            print("You can still test the API endpoints\n")
        else:
            print("[OK] All credentials configured")
    except Exception as e:
        print(f"[WARNING] Configuration issue: {e}")
    
    print("Starting FastAPI application")
    print(f"API running on http://localhost:{settings.port}")
    print(f"Health check: http://localhost:{settings.port}/health")
    print(f"Test endpoint: http://localhost:{settings.port}/api/analysis/test")
    print(f"Swagger UI: http://localhost:{settings.port}/api/docs")
    print(f"ReDoc: http://localhost:{settings.port}/api/redoc")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    )
