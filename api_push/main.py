from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime
from typing import Any

# Local imports
from config import settings
from auth import verify_bearer_token
from models import APIResponse, PushPayload, BusOccupancyData, HealthCheck
from database import supabase_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Webhook notification function
async def notify_frontend_of_new_data(record_id, data):
    """
    Send webhook notification to frontend deployment about new detection data
    """
    import httpx
    
    # Configure your frontend webhook URL
    FRONTEND_WEBHOOK_URL = os.getenv("FRONTEND_WEBHOOK_URL", "https://your-frontend-deployment.vercel.app/api/webhook/new-data")
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")
    
    try:
        webhook_payload = {
            "event": "new_detection_data",
            "record_id": record_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "secret": WEBHOOK_SECRET
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                FRONTEND_WEBHOOK_URL,
                json=webhook_payload,
                timeout=10.0
            )
            
        if response.status_code == 200:
            logger.info(f"Webhook sent successfully to frontend for record {record_id}")
        else:
            logger.warning(f"Webhook failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        logger.error(f"Error sending webhook to frontend: {e}")
        raise

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for handling push requests with bearer token authentication",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return APIResponse(
        success=True,
        message="Bus Occupancy API is running",
        data={"version": "1.0.0", "environment": settings.ENVIRONMENT}
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_connected = True
        try:
            supabase_client.get_client()
        except Exception:
            db_connected = False
        
        return HealthCheck(
            status="healthy" if db_connected else "degraded",
            database_connected=db_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post(f"{settings.API_V1_PREFIX}/push", response_model=APIResponse)
async def push_data(
    payload: PushPayload,
    token: str = Depends(verify_bearer_token)
):
    try:
        logger.info(f"Received push request with token: {token[:10]}...")

        import uuid

        # Log the received UUID for debugging
        logger.info(f"Received UUID from payload: {payload.uuid}")
        
        # Use existing UUID or generate a new one
        record_uuid = payload.uuid or str(uuid.uuid4())
        
        # Log what UUID we're actually using
        if payload.uuid:
            logger.info(f"Using provided UUID: {record_uuid}")
        else:
            logger.info(f"No UUID provided, generated new one: {record_uuid}")

        # Use `payload.data` if it exists; otherwise use the full model dict
        if payload.data:
            data_content = payload.data
        else:
            data_content = payload.model_dump(exclude_none=True, exclude={"metadata"})

        json_data = {
            "uuid": record_uuid,
            "received_at": datetime.utcnow().isoformat(),
            "data": data_content,
            "metadata": payload.metadata or {},
            "processed": True
        }

        try:
            result = await supabase_client.insert_data("push_requests", json_data)
            logger.info(f"Data stored successfully: {result}")

            if "detection_results" in data_content:
                try:
                    await notify_frontend_of_new_data(
                        record_id=result.data[0]["id"],
                        data={**data_content, "uuid": record_uuid}
                    )
                    logger.info(f"Frontend notification sent successfully for UUID {record_uuid}")
                except Exception as webhook_error:
                    logger.error(f"Frontend notification error: {webhook_error}")

        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            json_data["database_error"] = str(db_error)

        return APIResponse(
            success=True,
            message="Data processed successfully",
            data={
                "processed_data": json_data,
                "payload_size": len(str(data_content))
            }
        )

    except Exception as e:
        logger.error(f"Error processing push request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )




@app.post(f"{settings.API_V1_PREFIX}/bus-occupancy", response_model=APIResponse)
async def update_bus_occupancy(
    occupancy_data: BusOccupancyData,
    token: str = Depends(verify_bearer_token)
):
    """
    Update bus occupancy data
    Specific endpoint for bus occupancy system
    """
    try:
        logger.info(f"Received bus occupancy update for bus {occupancy_data.bus_id}")
        
        # Prepare JSON data for simplified schema
        json_data = {
            "bus_id": occupancy_data.bus_id,
            "route_id": occupancy_data.route_id,
            "occupancy_count": occupancy_data.occupancy_count,
            "max_capacity": occupancy_data.max_capacity,
            "occupancy_percentage": occupancy_data.occupancy_percentage,
            "timestamp": occupancy_data.timestamp.isoformat(),
            "location": occupancy_data.location
        }
        
        # Store in database (simplified schema: id, created_at, json_data)
        try:
            result = await supabase_client.insert_data("bus_occupancy", json_data)
            logger.info(f"Bus occupancy data stored successfully")
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store occupancy data"
            )
        
        return APIResponse(
            success=True,
            message="Bus occupancy updated successfully",
            data={
                "bus_id": occupancy_data.bus_id,
                "occupancy_percentage": occupancy_data.occupancy_percentage,
                "timestamp": occupancy_data.timestamp
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating bus occupancy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update occupancy: {str(e)}"
        )

@app.get(f"{settings.API_V1_PREFIX}/bus-occupancy/{{bus_id}}", response_model=APIResponse)
async def get_bus_occupancy(
    bus_id: str,
    token: str = Depends(verify_bearer_token)
):
    """
    Get current bus occupancy data for a specific bus
    """
    try:
        logger.info(f"Retrieving occupancy data for bus {bus_id}")
        
        # Get latest data for the bus from simplified schema
        result = await supabase_client.get_data("bus_occupancy")
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No occupancy data found for bus {bus_id}"
            )
        
        # Filter and get the most recent entry for the specific bus
        bus_data = [
            entry for entry in result.data 
            if entry.get('json_data', {}).get('bus_id') == bus_id
        ]
        
        if not bus_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No occupancy data found for bus {bus_id}"
            )
        
        # Get the most recent entry based on created_at
        latest_data = max(bus_data, key=lambda x: x.get('created_at', ''))
        
        return APIResponse(
            success=True,
            message="Bus occupancy data retrieved successfully",
            data=latest_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving bus occupancy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve occupancy data: {str(e)}"
        )

# Exception handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=APIResponse(
            success=False,
            message="Validation error",
            data={"detail": exc.detail if hasattr(exc, 'detail') else str(exc)}
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )
