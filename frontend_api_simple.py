"""
Simplified Frontend API for debugging
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bus Occupancy Frontend API",
    description="Frontend API for serving seating layout data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Webhook payload model
class WebhookPayload(BaseModel):
    event: str
    record_id: int
    data: Dict[str, Any]
    timestamp: str
    secret: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Bus Occupancy Frontend API", "status": "running", "version": "simple"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/webhook/new-data")
async def webhook_new_data(payload: WebhookPayload):
    """
    Webhook endpoint to receive notifications about new detection data
    """
    try:
        # Verify webhook secret
        expected_secret = os.getenv("WEBHOOK_SECRET", "bus-webhook-secret-2025")
        if payload.secret != expected_secret:
            logger.warning("Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
        logger.info(f"Received webhook for record {payload.record_id}")
        
        # For now, just return success without processing
        return {
            "success": True,
            "message": "Webhook received successfully",
            "record_id": payload.record_id,
            "event": payload.event
        }
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

@app.get("/api/seating-layout")
async def get_seating_layout():
    """
    Get the current seating layout (public endpoint)
    """
    try:
        # Return dummy data for now
        dummy_layout = {
            "Row1": {
                "A": {"class_id": 0, "class_name": "occupied", "confidence": 0.95},
                "B": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.90}
            },
            "Row2": {
                "A": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.88},
                "B": {"class_id": 0, "class_name": "occupied", "confidence": 0.92}
            }
        }
        
        return {
            "success": True,
            "data": {
                "layout": dummy_layout,
                "updated_at": datetime.utcnow().isoformat(),
                "source": "dummy_data"
            }
        }
            
    except Exception as e:
        logger.error(f"Error retrieving seating layout: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving seating layout: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
