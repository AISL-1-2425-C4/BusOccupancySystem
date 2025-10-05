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
import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. https://abcxyz.supabase.co
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role key

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
    data: Dict[str, Any] = None  # This contains detection_results and other data
    timestamp: str
    secret: str

# Global variable to store latest seating layout (in-memory cache)
latest_seating_layout = None
last_updated = None
previous_processed_layouts: list = []   # 👈 add this

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Bus Occupancy Frontend API", "status": "running", "version": "simple"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

async def get_last_five_excluding_latest():
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/push_requests",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            },
            params={
                "select": "id,created_at,json_data",
                "order": "id.desc",
                "limit": 5  # latest + 5 before it
            }
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase error: {resp.text}")

        rows = resp.json()
        # skip latest, return next 5, and include json_data in the output
        last_five = rows[1:]
        # Return id, created_at, and json_data for each
        return [
            {
                "id": r.get("id"),
                "created_at": r.get("created_at"),
                "json_data": r.get("json_data")
            }
            for r in last_five
        ]

#obtain 5 data entries from db



@app.post("/api/webhook/new-data")
async def webhook_new_data(payload: WebhookPayload):
    """
    Webhook endpoint to receive notifications about new detection data
    """
    response = {}

    try:
        # Verify webhook secret
        expected_secret = os.getenv("WEBHOOK_SECRET", "your-secure-webhook-secret-123")
        if payload.secret != expected_secret:
            logger.warning("Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
        logger.info(f"Received webhook for record {payload.record_id}")

        seating_layout = None

        # Access detection_results from the nested data structure
        detection_results = None
        if payload.data and "detection_results" in payload.data:
            detection_results = payload.data["detection_results"]
            logger.info(f"Found detection_results in payload.data: {len(detection_results)} detections")
        
        if detection_results is not None:
            logger.info(f"Processing {len(detection_results)} detection results")

            try:
                from seating_processor import process_seating_layout
                seating_layout = process_seating_layout(detection_results)
                if seating_layout:
                    logger.info(f"Successfully processed {len(detection_results)} detections using full seating.py algorithm")
                else:
                    logger.warning("seating_processor returned no layout")
            except ImportError as e:
                logger.error(f"Could not import seating_processor: {e}")
                # Fallback to simple processing
                seating_layout = process_detection_results_simple(detection_results)
            except Exception as e:
                logger.error(f"Error in seating_processor: {e}")
                # Fallback to simple processing
                seating_layout = process_detection_results_simple(detection_results)

            if seating_layout:
                global latest_seating_layout, last_updated, previous_processed_layouts
                latest_seating_layout = seating_layout
                last_updated = datetime.utcnow().isoformat()
                from uuid import uuid4

                # === Save processed layout to Supabase (different table) ===
                try:
                    unique_id = str(uuid4())
                    created_at = datetime.utcnow().isoformat()

                    processed_record = {
                        "uuid": unique_id,
                        "created_at": created_at,
                        "record_id": payload.record_id,
                        "layout_data": seating_layout,       # You could also store 'final_layout' later if preferred
                        "source": "webhook_new_data",
                    }

                    SUPABASE_PROCESSED_TABLE = os.getenv("SUPABASE_PROCESSED_TABLE", "processed_layouts")

                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"{SUPABASE_URL}/rest/v1/{SUPABASE_PROCESSED_TABLE}",
                            headers={
                                "apikey": SUPABASE_KEY,
                                "Authorization": f"Bearer {SUPABASE_KEY}",
                                "Content-Type": "application/json",
                                "Prefer": "return=representation"
                            },
                            json=processed_record
                        )

                    if resp.status_code in [200, 201]:
                        inserted = resp.json()[0]
                        logger.info(
                            f"✅ Successfully saved processed layout to Supabase table '{SUPABASE_PROCESSED_TABLE}' "
                            f"(UUID={unique_id}, RecordID={payload.record_id})"
                        )
                    else:
                        logger.error(
                            f"❌ Failed to insert into '{SUPABASE_PROCESSED_TABLE}': {resp.status_code} {resp.text}"
                        )

                except Exception as e:
                    logger.error(f"🔥 Error saving processed layout to Supabase: {e}")




                # Fetch the last 4 raw layouts from Supabase
                try:
                    last_four_raw = await get_last_five_excluding_latest()

                    # Process them before returning
                    previous_processed_layouts = []
                    for r in last_four_raw:
                        json_data = r.get("json_data", {})
                        detections = json_data.get("detection_results", [])
                        if detections:
                            processed = process_seating_layout(detections)
                            if processed:
                                previous_processed_layouts.append({
                                    "record_id": r["id"],  # keep track of which record
                                    "layout": processed
                                })


                    response["previous_layouts"] = previous_processed_layouts

                except Exception as e:
                    logger.error(f"Error fetching/processing last four layouts: {e}")
                    response["previous_layouts"] = []
                    previous_processed_layouts = []  # reset if failed

                # Combine latest + previous 4 into one majority-voted layout
                all_layouts = [latest_seating_layout] + [p["layout"] for p in previous_processed_layouts]
                final_layout = merge_seating_layouts(all_layouts)

                # Save the final merged layout into cache
                latest_seating_layout = final_layout

                response = {
                    "success": True,
                    "message": "Detection data processed and merged (majority vote)",
                    "record_id": payload.record_id,
                    "rows_processed": len(final_layout),
                    "updated_at": last_updated,
                    "latest_layout": final_layout,                    # 👈 merged layout (frontend will use this)
                    "previous_layouts": previous_processed_layouts,   # 👈 still keep raw 4 older ones
                }



            else:
                response = {
                    "success": False,
                    "message": "Failed to process seating layout",
                    "record_id": payload.record_id
                }
        else:
            logger.info("No detection_results in webhook payload")
            logger.info(f"Payload data structure: {payload.data}")
            response = {
                "success": True,
                "message": "No detection data to process",
                "record_id": payload.record_id,
                "payload_data": payload.data
            }

        # ✅ Add last 5 records from DB (excluding latest)
        try:
            last_five = await get_last_five_excluding_latest()
            response["last_five_records"] = last_five

            # Log IDs and timestamps for clarity
            logger.info(
                f"Fetched {len(last_five)} previous records: "
                + ", ".join([f"id={r['id']} created_at={r['created_at']}" for r in last_five])
            )

        except Exception as e:
            logger.error(f"Error fetching last five records: {e}")
            response["last_five_records"] = []


    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

    return response

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

@app.get("/availability.html")
async def serve_availability():
    if os.path.exists("availability.html"):
        return FileResponse("availability.html")
    raise HTTPException(status_code=404, detail="Availability page not found")

# Serve CSS files
@app.get("/style.css")
async def serve_css():
    """Serve main CSS file"""
    if os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

# Serve JavaScript files
@app.get("/auto_refresh_seating.js")
async def serve_auto_refresh_js():
    """Serve auto-refresh JavaScript file"""
    if os.path.exists("auto_refresh_seating.js"):
        return FileResponse("auto_refresh_seating.js", media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")
from collections import Counter

def merge_seating_layouts(layouts: list[dict]) -> dict:
    """
    Merge multiple seating layouts by majority vote on class_id.
    Keeps coordinates from the most recent layout.
    """
    if not layouts:
        return {}

    # Use the most recent layout as the base (so we keep coords/structure)
    base_layout = layouts[0]  

    merged = {}

    for row, cols in base_layout.items():
        merged[row] = {}
        for col, seat_info in cols.items():
            # Collect all class_id values for this seat across all layouts
            class_ids = []
            coords = seat_info.get("coordinates", {})
            for layout in layouts:
                try:
                    seat = layout[row][col]
                    class_ids.append(seat["class_id"])
                except KeyError:
                    continue  # skip if missing in that layout

            if class_ids:
                # Majority vote
                most_common = Counter(class_ids).most_common(1)[0][0]
            else:
                most_common = seat_info["class_id"]  # fallback

            merged[row][col] = {
                "class_id": most_common,
                "coordinates": coords,  # preserve coords from latest
            }

    return merged

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

def process_detection_results_full(detection_results):
    """
    Process detection results using the actual seating.py algorithm
    """
    try:
        if not detection_results:
            return None
            
        logger.info(f"Processing {len(detection_results)} detection results using seating.py algorithm")
        
        # Import seating processing functions
        from seating_lib import process_detections_to_layout
        
        # Use the actual seating.py logic
        seating_layout = process_detections_to_layout(detection_results)
        
        if seating_layout:
            logger.info(f"Generated seating layout with {len(seating_layout)} rows using seating.py algorithm")
        else:
            logger.warning("seating.py algorithm returned no layout")
            
        return seating_layout
        
    except ImportError as e:
        logger.error(f"Could not import seating_lib: {e}")
        # Fallback to simplified processing
        return process_detection_results_simple(detection_results)
    except Exception as e:
        logger.error(f"Error in process_detection_results_full: {e}")
        return None


def process_detection_results_simple(detection_results):
    """
    Simplified fallback processing (original logic)
    """
    try:
        if not detection_results:
            return None
            
        logger.info(f"Using fallback simple processing for {len(detection_results)} detections")
        
        # Create the standard 42-seat bus layout structure
        seating_layout = {}
        
        # Define the standard bus seating arrangement (10 rows, varying columns)
        bus_layout = {
            "row_1": ["column_one", "column_two", "column_three", "column_four"],
            "row_2": ["column_one", "column_two", "column_three", "column_four"], 
            "row_3": ["column_one", "column_two", "column_three", "column_four"],
            "row_4": ["column_one", "column_two", "column_three", "column_four"],
            "row_5": ["column_one", "column_two", "column_three", "column_four"],
            "row_6": ["column_one", "column_two", "column_three", "column_four"],
            "row_7": ["column_one", "column_two", "column_three", "column_four"],
            "row_8": ["column_one", "column_two", "column_three", "column_four"],
            "row_9": ["column_one", "column_two", "column_three", "column_four"],
            "row_10": ["column_one", "column_two", "column_three", "column_four", "column_five", "column_six"]
        }
        
        # Initialize all seats as unoccupied by default
        for row_name, columns in bus_layout.items():
            seating_layout[row_name] = {}
            for i, col_name in enumerate(columns):
                # Calculate standard positions for the bus layout
                row_num = int(row_name.split('_')[1])
                col_num = i + 1
                
                # Standard positioning (matches your existing layout)
                x_pos = 30 + (row_num - 1) * 55  # Rows go from left to right
                if col_num <= 2:  # Left side seats
                    y_pos = 25 + (col_num - 1) * 55
                else:  # Right side seats (after aisle)
                    y_pos = 245 + (col_num - 3) * 55
                
                seating_layout[row_name][col_name] = {
                    "class_id": 1,  # Default: unoccupied
                    "class_name": "unoccupied",
                    "confidence": 0.5,
                    "position": {"x": float(x_pos), "y": float(y_pos)}
                }
        
        # Map detections to seats (simplified proximity matching)
        for detection in detection_results:
            x_min, y_min = detection["x_min"], detection["y_min"] 
            x_max, y_max = detection["x_max"], detection["y_max"]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Find the closest seat to this detection
            min_distance = float('inf')
            closest_seat = None
            
            for row_name, row_data in seating_layout.items():
                for col_name, seat_data in row_data.items():
                    seat_x = seat_data["position"]["x"]
                    seat_y = seat_data["position"]["y"]
                    
                    # Scale detection coordinates to match seat layout coordinates
                    scaled_x = (x_center / 4608) * 660
                    scaled_y = (y_center / 2592) * 405
                    
                    distance = ((scaled_x - seat_x) ** 2 + (scaled_y - seat_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_seat = (row_name, col_name)
            
            # Update the closest seat with detection data
            if closest_seat and min_distance < 100:
                row_name, col_name = closest_seat
                
                # Determine occupancy status
                class_id = detection.get("class_id", 0)
                class_name = detection.get("class_name", "occupied").lower()
                
                if class_id == 0 or "occupied" in class_name:
                    final_class_id = 0
                    final_class_name = "occupied"
                else:
                    final_class_id = 1
                    final_class_name = "unoccupied"
                
                seating_layout[row_name][col_name].update({
                    "class_id": final_class_id,
                    "class_name": final_class_name,
                    "confidence": detection.get("confidence", 0.5)
                })
        
        logger.info(f"Generated seating layout with {len(seating_layout)} rows using simple processing")
        return seating_layout
        
    except Exception as e:
        logger.error(f"Error in process_detection_results_simple: {e}")
        return None


def process_detection_results(detection_results):
    """
    Process detection results into seating layout (simplified version - DEPRECATED)
    """
    # Use the full version instead
    return process_detection_results_full(detection_results)
    #         return 3

    #     def find_matching_detection(x, y, detections, tolerance=50):
    #         for detection in detections:
    #             det_x_center = (detection["x_min"] + detection["x_max"]) / 2
    #             det_y_center = (detection["y_min"] + detection["y_max"]) / 2
    #             if abs(det_x_center - x) < tolerance and abs(det_y_center - y) < tolerance:
    #                 return detection
    #         return None

    #     # Create row-based seating data
    #     row_data = {}
    #     for row_idx, row in enumerate(rows):
    #         row_name = f"Row{row_idx + 1}"
    #         row_seats = {}
            
    #         for seat_idx, (x, y) in enumerate(row):
    #             col = assign_to_column(x)
    #             col_name = ["A", "B", "C", "D"][col] if col < 4 else f"Col{col}"
                
    #             detection = find_matching_detection(x, y, detection_results)
    #             if detection:
    #                 row_seats[col_name] = {
    #                     "class_id": detection["class_id"],
    #                     "class_name": detection.get("class_name", "Unknown"),
    #                     "confidence": detection["confidence"],
    #                     "position": {"x": x, "y": y}
    #                 }
            
    #         if row_seats:
    #             row_data[row_name] = row_seats

    #     logger.info(f"Processed {len(row_data)} rows from {len(detection_results)} detections")
    #     return row_data
        
    # except Exception as e:
    #     logger.error(f"Error processing detection results: {e}")
    #     return None

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
