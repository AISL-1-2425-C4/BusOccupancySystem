"""
Frontend API for Bus Occupancy System
Separate deployment that handles seating layout processing and serves data to HTML
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel
import subprocess
import sys

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

# Global variable to store latest seating layout
latest_seating_layout = None
last_updated = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Bus Occupancy Frontend API", "status": "running"}

@app.post("/api/webhook/new-data")
async def webhook_new_data(payload: WebhookPayload):
    """
    Webhook endpoint to receive notifications about new detection data
    """
    global latest_seating_layout, last_updated
    
    try:
        # Verify webhook secret
        expected_secret = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")
        if payload.secret != expected_secret:
            logger.warning("Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
        logger.info(f"Received webhook for record {payload.record_id}")
        
        # Check if this contains detection_results
        if "detection_results" in payload.data:
            detection_results = payload.data["detection_results"]
            
            # Process the seating layout
            seating_layout = process_detection_results(detection_results)
            
            if seating_layout:
                # Update global variables
                latest_seating_layout = seating_layout
                last_updated = datetime.utcnow().isoformat()
                
                # Save to JSON file for HTML compatibility
                save_seating_layout_to_file(seating_layout)
                
                logger.info(f"Processed seating layout with {len(seating_layout)} rows")
                
                return {
                    "success": True,
                    "message": "Detection data processed successfully",
                    "record_id": payload.record_id,
                    "rows_processed": len(seating_layout)
                }
            else:
                logger.warning("Failed to process seating layout")
                return {
                    "success": False,
                    "message": "Failed to process seating layout",
                    "record_id": payload.record_id
                }
        else:
            logger.info("No detection_results in webhook payload")
            return {
                "success": True,
                "message": "No detection data to process",
                "record_id": payload.record_id
            }
            
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

@app.get("/api/seating-layout")
async def get_seating_layout():
    """
    Get the current seating layout (public endpoint)
    """
    global latest_seating_layout, last_updated
    
    try:
        # If we have cached data, return it
        if latest_seating_layout:
            return {
                "success": True,
                "data": {
                    "layout": latest_seating_layout,
                    "updated_at": last_updated,
                    "source": "webhook_processed"
                }
            }
        
        # Fallback: try to read from JSON file
        try:
            with open("row_seating_layout.json", "r") as f:
                layout_data = json.load(f)
            
            return {
                "success": True,
                "data": {
                    "layout": layout_data,
                    "updated_at": None,
                    "source": "json_file"
                }
            }
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="No seating layout data available"
            )
            
    except Exception as e:
        logger.error(f"Error retrieving seating layout: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving seating layout: {str(e)}"
        )

@app.post("/api/process-manual")
async def process_manual():
    """
    Manually trigger processing using the local seating.py script
    """
    try:
        logger.info("Manually triggering seating.py processing")
        
        # Run seating.py script
        result = subprocess.run(
            [sys.executable, "seating.py", "--supabase"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            # Try to load the generated JSON
            try:
                with open("row_seating_layout.json", "r") as f:
                    layout_data = json.load(f)
                
                # Update global cache
                global latest_seating_layout, last_updated
                latest_seating_layout = layout_data
                last_updated = datetime.utcnow().isoformat()
                
                return {
                    "success": True,
                    "message": "Seating layout processed successfully",
                    "output": result.stdout,
                    "layout": layout_data
                }
            except Exception as file_error:
                return {
                    "success": True,
                    "message": "Processing completed but couldn't load JSON",
                    "output": result.stdout,
                    "error": str(file_error)
                }
        else:
            logger.error(f"seating.py failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.stderr}"
            )
            
    except Exception as e:
        logger.error(f"Error in manual processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in manual processing: {str(e)}"
        )

def process_detection_results(detection_results):
    """
    Process detection results into seating layout (simplified version)
    """
    try:
        # Extract midpoints from detections
        midpoints = []
        for detection in detection_results:
            x_min, y_min = detection["x_min"], detection["y_min"]
            x_max, y_max = detection["x_max"], detection["y_max"]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            midpoints.append((x_center, y_center))

        # Sort midpoints by y (top to bottom)
        midpoints.sort(key=lambda pt: pt[1])

        # Group midpoints into rows by y proximity
        row_threshold = 60
        rows = []
        for pt in midpoints:
            placed = False
            for row in rows:
                if abs(row[0][1] - pt[1]) < row_threshold:
                    row.append(pt)
                    placed = True
                    break
            if not placed:
                rows.append([pt])

        # Sort each row by x (left to right)
        for row in rows:
            row.sort(key=lambda pt: pt[0])

        # Create seating layout
        canvas_width = 4608
        col_edges = [0, canvas_width/4, canvas_width/2, 3*canvas_width/4, canvas_width]
        
        def assign_to_column(x):
            for i in range(4):
                if col_edges[i] <= x < col_edges[i+1]:
                    return i
            return 3

        def find_matching_detection(x, y, detections, tolerance=50):
            for detection in detections:
                det_x_center = (detection["x_min"] + detection["x_max"]) / 2
                det_y_center = (detection["y_min"] + detection["y_max"]) / 2
                if abs(det_x_center - x) < tolerance and abs(det_y_center - y) < tolerance:
                    return detection
            return None

        # Create row-based seating data
        row_data = {}
        for row_idx, row in enumerate(rows):
            row_name = f"Row{row_idx + 1}"
            row_seats = {}
            
            for seat_idx, (x, y) in enumerate(row):
                col = assign_to_column(x)
                col_name = ["A", "B", "C", "D"][col] if col < 4 else f"Col{col}"
                
                detection = find_matching_detection(x, y, detection_results)
                if detection:
                    row_seats[col_name] = {
                        "class_id": detection["class_id"],
                        "class_name": detection.get("class_name", "Unknown"),
                        "confidence": detection["confidence"],
                        "position": {"x": x, "y": y}
                    }
            
            if row_seats:
                row_data[row_name] = row_seats

        return row_data
        
    except Exception as e:
        logger.error(f"Error processing detection results: {e}")
        return None

def save_seating_layout_to_file(layout_data, filename="row_seating_layout.json"):
    """
    Save seating layout to JSON file
    """
    try:
        with open(filename, "w") as f:
            json.dump(layout_data, f, indent=2)
        logger.info(f"Saved seating layout to {filename}")
    except Exception as e:
        logger.error(f"Error saving layout to file: {e}")

# Serve static files (HTML, CSS, JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main HTML file
@app.get("/bus-layout")
async def serve_bus_layout():
    """Serve the main bus layout HTML"""
    html_files = ["dynamic_seating.html", "newmorning.html", "newafternoon.html"]
    
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    
    raise HTTPException(status_code=404, detail="Bus layout HTML not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
