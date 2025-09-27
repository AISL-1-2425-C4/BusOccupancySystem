"""
Simplified Frontend API for debugging
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# Global variable to store latest seating layout (in-memory cache)
latest_seating_layout = None
last_updated = None

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
        expected_secret = os.getenv("WEBHOOK_SECRET", "your-secure-webhook-secret-123")
        if payload.secret != expected_secret:
            logger.warning("Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
        logger.info(f"Received webhook for record {payload.record_id}")
        
        # Check if this contains detection_results
        if "detection_results" in payload.data:
            detection_results = payload.data["detection_results"]
            logger.info(f"Processing {len(detection_results)} detection results")
            
            # Process the seating layout
            seating_layout = process_detection_results(detection_results)
            
            if seating_layout:
                # Store in memory (works in Vercel)
                global latest_seating_layout, last_updated
                latest_seating_layout = seating_layout
                last_updated = datetime.utcnow().isoformat()
                
                # Try to save to file (may not persist in Vercel, but worth trying)
                save_seating_layout_to_file(seating_layout)
                
                return {
                    "success": True,
                    "message": "Detection data processed and cached",
                    "record_id": payload.record_id,
                    "rows_processed": len(seating_layout),
                    "updated_at": last_updated
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

# Serve static files (images, CSS, JS) - MUST be before HTML routes
if os.path.exists("images"):
    app.mount("/images", StaticFiles(directory="images"), name="images")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML files based on newhome.html navigation
@app.get("/newhome.html")
@app.get("/index")
@app.get("/home")
async def serve_home():
    """Serve the main home page"""
    if os.path.exists("newhome.html"):
        return FileResponse("newhome.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="Home page not found")

@app.get("/newmorning.html")
@app.get("/morning")
async def serve_morning():
    """Serve morning HTML page"""
    html_files = ["newmorning.html", "morning.html"]
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Morning page not found")

@app.get("/newafternoon.html")
@app.get("/afternoon")
async def serve_afternoon():
    """Serve afternoon HTML page"""
    html_files = ["newafternoon.html", "afternoon.html"]
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Afternoon page not found")

@app.get("/aboutus.html")
async def serve_about():
    """Serve about us page"""
    if os.path.exists("aboutus.html"):
        return FileResponse("aboutus.html")
    raise HTTPException(status_code=404, detail="About page not found")

@app.get("/routes.html")
async def serve_routes():
    """Serve routes and schedules page"""
    if os.path.exists("routes.html"):
        return FileResponse("routes.html")
    raise HTTPException(status_code=404, detail="Routes page not found")

@app.get("/prob.html")
async def serve_contact():
    """Serve contact us page"""
    if os.path.exists("prob.html"):
        return FileResponse("prob.html")
    raise HTTPException(status_code=404, detail="Contact page not found")

@app.get("/bus-layout")
@app.get("/dynamic_seating.html")
async def serve_bus_layout():
    """Serve the main bus layout HTML"""
    html_files = ["dynamic_seating.html", "seating.html"]
    for html_file in html_files:
        if os.path.exists(html_file):
            return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Bus layout HTML not found")

# Serve CSS files
@app.get("/style.css")
async def serve_css():
    """Serve main CSS file"""
    if os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

# Serve JSON files for backward compatibility
@app.get("/row_seating_layout.json")
async def serve_seating_json():
    """Serve seating layout JSON - Vercel-compatible version"""
    try:
        global latest_seating_layout, last_updated
        
        # First try to return cached data from memory
        if latest_seating_layout:
            logger.info("Serving seating layout from memory cache")
            return latest_seating_layout
        
        # Try to serve existing JSON file (if it exists)
        if os.path.exists("row_seating_layout.json"):
            logger.info("Serving seating layout from file")
            return FileResponse("row_seating_layout.json", media_type="application/json")
        
        # Fallback: return dummy data
        logger.info("Serving dummy seating layout")
        dummy_layout = {
            "Row1": {
                "A": {"class_id": 0, "class_name": "occupied", "confidence": 0.95},
                "B": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.90}
            },
            "Row2": {
                "A": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.88},
                "B": {"class_id": 0, "class_name": "occupied", "confidence": 0.92}
            },
            "Row3": {
                "A": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.85},
                "B": {"class_id": 1, "class_name": "unoccupied", "confidence": 0.88}
            }
        }
        
        return dummy_layout
        
    except Exception as e:
        logger.error(f"Error serving seating JSON: {e}")
        raise HTTPException(status_code=404, detail="Seating layout not found")

# Serve other JSON files that might be needed
@app.get("/seat_mapping.json")
async def serve_seat_mapping():
    """Serve seat mapping JSON file"""
    if os.path.exists("seat_mapping.json"):
        return FileResponse("seat_mapping.json", media_type="application/json")
    raise HTTPException(status_code=404, detail="Seat mapping not found")

def process_detection_results(detection_results):
    """
    Process detection results into seating layout (simplified version)
    """
    try:
        if not detection_results:
            return None
            
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

        logger.info(f"Processed {len(row_data)} rows from {len(detection_results)} detections")
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
        return True
    except Exception as e:
        logger.error(f"Error saving layout to file: {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
