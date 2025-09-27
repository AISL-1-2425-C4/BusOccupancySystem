# 🚀 Separate Deployment Architecture Guide

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    Webhook    ┌──────────────────┐    API Calls    ┌─────────────┐
│   Detection     │──────────────▶│    Frontend      │◀────────────────│   HTML      │
│   API (Vercel)  │               │   API (Vercel)   │                 │  (Browser)  │
│                 │               │                  │                 │             │
│ - Receives data │               │ - Processes data │                 │ - Displays  │
│ - Stores in DB  │               │ - Serves layout  │                 │   layout    │
│ - Sends webhook │               │ - Handles HTML   │                 │ - Polls API │
└─────────────────┘               └──────────────────┘                 └─────────────┘
        │                                   │
        │                                   │
        └───────────────┐         ┌─────────┘
                        ▼         ▼
                   ┌─────────────────┐
                   │   Supabase DB   │
                   │                 │
                   │ - push_requests │
                   │ - bus_seating   │
                   └─────────────────┘
```

## 📂 **Deployment Structure**

### **Deployment 1: Detection API** (`api-push-2oek.vercel.app`)
- **Purpose**: Receives detection data, stores in Supabase, sends webhooks
- **Files**: `api_push/` folder
- **Responsibilities**:
  - Receive POST requests with detection data
  - Store data in `push_requests` table
  - Send webhook notifications to frontend
  - Provide health checks

### **Deployment 2: Frontend API** (`bus-frontend-xyz.vercel.app`)
- **Purpose**: Processes seating layouts, serves HTML, provides data API
- **Files**: Root folder + `frontend_api.py`
- **Responsibilities**:
  - Receive webhook notifications
  - Process detection data into seating layouts
  - Serve seating data via API
  - Host HTML files
  - Handle manual processing requests

## 🔧 **Setup Instructions**

### **Step 1: Deploy Detection API (Current Setup)**

Your existing detection API is already working. Just add these environment variables in Vercel:

```
FRONTEND_WEBHOOK_URL=https://your-frontend-deployment.vercel.app/api/webhook/new-data
WEBHOOK_SECRET=your-secure-webhook-secret-here
```

### **Step 2: Deploy Frontend API**

1. **Create new Vercel project** for frontend
2. **Copy these files** to the new project:
   ```
   frontend_api.py
   seating.py
   requirements.txt (create new one)
   frontend_vercel.json (rename to vercel.json)
   dynamic_seating.html (or your HTML files)
   row_seating_layout.json (if exists)
   ```

3. **Create requirements.txt for frontend**:
   ```
   fastapi==0.104.1
   python-multipart==0.0.6
   supabase==2.0.2
   python-dotenv==1.0.0
   pydantic==2.5.0
   httpx>=0.24.0,<0.25.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   ```

4. **Set environment variables** in Vercel:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   WEBHOOK_SECRET=your-secure-webhook-secret-here
   ENVIRONMENT=production
   ```

### **Step 3: Update HTML Files**

Add the JavaScript integration to your HTML files:

```html
<!-- Add before closing </body> tag -->
<script src="html_api_integration.js"></script>
<script>
// Initialize with your frontend deployment URL
const busAPI = new BusSeatingAPI('https://your-frontend-deployment.vercel.app');

// Replace your existing data loading with API calls
document.addEventListener('DOMContentLoaded', function() {
    // Start automatic polling
    busAPI.startPolling();
});
</script>
```

## 🔄 **Data Flow**

### **Webhook Flow (Real-time)**
1. Detection system sends data to: `api-push-2oek.vercel.app/api/v1/push`
2. Detection API stores data in Supabase `push_requests` table
3. Detection API sends webhook to: `frontend-deployment.vercel.app/api/webhook/new-data`
4. Frontend API processes detection data into seating layout
5. Frontend API updates cached layout and saves JSON file
6. HTML polls frontend API and gets updated layout

### **Polling Flow (Fallback)**
1. HTML polls: `frontend-deployment.vercel.app/api/seating-layout` every 30 seconds
2. Frontend API returns cached layout or processes latest data from Supabase
3. HTML updates display if new data is available

## 🧪 **Testing the Setup**

### **1. Test Detection API**
```bash
curl -X POST "https://api-push-2oek.vercel.app/api/v1/push" \
  -H "Authorization: Bearer Y7a450d69-8ef6-4249-87cf-70cf7ce0d621" \
  -H "Content-Type: application/json" \
  -d '{"data": {"detection_results": [{"image": "test.jpg", "class_id": 1, "class_name": "unoccupied", "confidence": 0.95, "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}]}}'
```

### **2. Test Frontend Webhook** (should happen automatically)
Check frontend logs for webhook processing messages.

### **3. Test Frontend API**
```bash
curl "https://your-frontend-deployment.vercel.app/api/seating-layout"
```

### **4. Test Manual Processing**
```bash
curl -X POST "https://your-frontend-deployment.vercel.app/api/process-manual"
```

### **5. Test HTML Integration**
Open your HTML page and check browser console for polling messages.

## 🔒 **Security Considerations**

1. **Webhook Secret**: Use a strong, unique secret for webhook verification
2. **Environment Variables**: Store all secrets in Vercel environment variables
3. **CORS**: Configure CORS properly for your domain
4. **Rate Limiting**: Consider adding rate limiting to public endpoints

## 📊 **Monitoring & Debugging**

### **Detection API Logs**
- Check Vercel function logs for webhook sending status
- Monitor Supabase for new `push_requests` records

### **Frontend API Logs**
- Check webhook reception and processing
- Monitor seating layout generation
- Watch for API polling requests

### **HTML Console**
- Check browser console for polling status
- Monitor network tab for API requests
- Look for seating layout update events

## 🚨 **Troubleshooting**

### **Webhook Not Received**
1. Check `FRONTEND_WEBHOOK_URL` environment variable
2. Verify webhook secret matches on both ends
3. Check frontend deployment logs
4. Test webhook endpoint manually

### **Seating Layout Not Updating**
1. Verify detection data contains `detection_results`
2. Check frontend processing logs
3. Test manual processing endpoint
4. Verify HTML polling is working

### **HTML Not Loading New Data**
1. Check browser console for errors
2. Verify API endpoint URLs
3. Test API endpoints manually
4. Check CORS configuration

## 🎯 **Benefits of This Architecture**

✅ **Separation of Concerns**: Detection API focuses on data ingestion, Frontend API focuses on processing and serving

✅ **Scalability**: Each service can be scaled independently

✅ **Reliability**: If one service fails, the other can continue operating

✅ **Flexibility**: Easy to modify frontend without affecting data ingestion

✅ **Real-time Updates**: Webhook system provides immediate updates

✅ **Fallback Mechanism**: Polling ensures updates even if webhooks fail

✅ **Easy Deployment**: Each service can be deployed and updated independently

## 🔄 **Alternative: Polling-Only Setup**

If you prefer to avoid webhooks, you can use polling-only:

1. Remove webhook code from detection API
2. Frontend API polls Supabase directly for new data
3. HTML continues to poll frontend API as before

This is simpler but less real-time than the webhook approach.
