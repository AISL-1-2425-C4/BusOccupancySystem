from picamera2 import Picamera2, Preview
import time
from datetime import datetime
from supabase import create_client, Client
import os
import uuid

# ------------------ SUPABASE DETAILS ------------------ #
SUPABASE_URL = "https://wzxwavklbsdmiwwbousk.storage.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6eHdhdmtsYnNkbWl3d2JvdXNrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc5ODY2NzgsImV4cCI6MjA3MzU2MjY3OH0.AvlTu4TIfvZW-mPz26rTwhITrbN5BBhWFqy_Dz22pO4"
BUCKET_NAME = "images-store"

# ------------------ INITIAL SETUP ------------------ #
# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Local directory for saving images
LOCAL_DIR = "/home/thesis/Deployment/DataCollection"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Initialize camera
camera = Picamera2()
config = camera.create_still_configuration(main={"size": (4068, 2592)})
camera.configure(config)
camera.start()
time.sleep(2)  # warm-up

# ------------------ FUNCTIONS ------------------ #
def capture_image():
    # Timestamp + UUID filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex  # generates unique 32-char hex string
    filename = f"{timestamp}-{unique_id}.jpg"

    local_path = os.path.join(LOCAL_DIR, filename)
    camera.capture_file(local_path)
    print(f"✅ Saved locally: {local_path}")
    return local_path, filename


def upload_to_supabase(file_path, filename, bucket=BUCKET_NAME):
    try:
        with open(file_path, "rb") as f:
            res = supabase.storage.from_(bucket).upload(filename, f, {"content-type": "image/jpeg"})
        print(f"☁️ Uploaded to Supabase: {filename}")
        return True
    except Exception as e:
        print(f"❌ Upload failed for {filename}: {e}")
        return False


# ------------------ MAIN LOOP ------------------ #
if __name__ == "__main__":
    try:
        while True:
            local_path, filename = capture_image()
            success = upload_to_supabase(local_path, filename)

            if success:
                print(f"✅ Both local save & cloud upload successful for {filename}")
            else:
                print(f"⚠️ Local save successful, but cloud upload failed for {filename}")

            time.sleep(2)  # capture every 2 seconds

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        camera.stop()
