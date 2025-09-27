/**
 * JavaScript code to integrate with the frontend API
 * Add this to your existing HTML files
 */

class BusSeatingAPI {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl || 'https://your-frontend-deployment.vercel.app';
        this.lastUpdated = null;
        this.pollInterval = 30000; // 30 seconds
        this.isPolling = false;
    }

    /**
     * Fetch the latest seating layout from the API
     */
    async fetchSeatingLayout() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/seating-layout`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                console.log('✅ Seating layout fetched successfully');
                return result.data;
            } else {
                throw new Error('API returned unsuccessful response');
            }
        } catch (error) {
            console.error('❌ Error fetching seating layout:', error);
            throw error;
        }
    }

    /**
     * Check if the seating layout has been updated
     */
    hasLayoutUpdated(newData) {
        if (!this.lastUpdated || !newData.updated_at) {
            return true; // First load or no timestamp
        }
        
        return new Date(newData.updated_at) > new Date(this.lastUpdated);
    }

    /**
     * Update the seating display with new data
     */
    updateSeatingDisplay(layoutData) {
        try {
            // Update your existing seating display logic here
            console.log('🔄 Updating seating display with new data');
            
            // Example: Update global variable if your HTML uses it
            if (typeof window.seatLayoutData !== 'undefined') {
                window.seatLayoutData = layoutData.layout;
            }
            
            // Example: Trigger your existing seat generation function
            if (typeof generateSeats === 'function') {
                generateSeats(layoutData.layout);
            }
            
            // Update last updated timestamp
            this.lastUpdated = layoutData.updated_at;
            
            // Update UI timestamp display
            this.updateTimestampDisplay(layoutData.updated_at);
            
            console.log('✅ Seating display updated successfully');
            
        } catch (error) {
            console.error('❌ Error updating seating display:', error);
        }
    }

    /**
     * Update timestamp display in the UI
     */
    updateTimestampDisplay(timestamp) {
        const timestampElement = document.getElementById('last-updated');
        if (timestampElement && timestamp) {
            const date = new Date(timestamp);
            timestampElement.textContent = `Last updated: ${date.toLocaleString()}`;
        }
    }

    /**
     * Start polling for updates
     */
    startPolling() {
        if (this.isPolling) {
            console.log('⚠️ Polling already started');
            return;
        }

        console.log(`🔄 Starting polling every ${this.pollInterval / 1000} seconds`);
        this.isPolling = true;

        // Initial load
        this.checkForUpdates();

        // Set up interval
        this.pollIntervalId = setInterval(() => {
            this.checkForUpdates();
        }, this.pollInterval);
    }

    /**
     * Stop polling for updates
     */
    stopPolling() {
        if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
        }
        this.isPolling = false;
        console.log('⏹️ Polling stopped');
    }

    /**
     * Check for updates and update display if needed
     */
    async checkForUpdates() {
        try {
            console.log('🔍 Checking for seating layout updates...');
            
            const layoutData = await this.fetchSeatingLayout();
            
            if (this.hasLayoutUpdated(layoutData)) {
                console.log('🆕 New seating data detected, updating display');
                this.updateSeatingDisplay(layoutData);
                
                // Trigger custom event for other parts of your app
                window.dispatchEvent(new CustomEvent('seatingLayoutUpdated', {
                    detail: layoutData
                }));
            } else {
                console.log('💤 No updates detected');
            }
            
        } catch (error) {
            console.error('❌ Error checking for updates:', error);
            
            // Fallback to local file if API fails
            this.fallbackToLocalFile();
        }
    }

    /**
     * Fallback to loading local JSON file if API fails
     */
    async fallbackToLocalFile() {
        try {
            console.log('🔄 Falling back to local JSON file...');
            
            const response = await fetch('row_seating_layout.json');
            if (response.ok) {
                const layoutData = await response.json();
                this.updateSeatingDisplay({ layout: layoutData, updated_at: null });
                console.log('✅ Loaded from local JSON file');
            }
        } catch (error) {
            console.error('❌ Fallback to local file also failed:', error);
        }
    }

    /**
     * Manually trigger processing (requires authentication)
     */
    async triggerManualProcessing() {
        try {
            console.log('🔄 Triggering manual processing...');
            
            const response = await fetch(`${this.apiBaseUrl}/api/process-manual`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Manual processing completed:', result);
                
                // Check for updates after processing
                setTimeout(() => this.checkForUpdates(), 2000);
                
                return result;
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Error triggering manual processing:', error);
            throw error;
        }
    }
}

// Usage example:
// Initialize the API client
const busAPI = new BusSeatingAPI('https://your-frontend-deployment.vercel.app');

// Start automatic polling when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Starting bus seating API integration');
    
    // Add timestamp display to your HTML
    const timestampDiv = document.createElement('div');
    timestampDiv.id = 'last-updated';
    timestampDiv.style.cssText = 'position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; z-index: 1000;';
    document.body.appendChild(timestampDiv);
    
    // Start polling
    busAPI.startPolling();
    
    // Add manual refresh button
    const refreshButton = document.createElement('button');
    refreshButton.textContent = '🔄 Refresh Data';
    refreshButton.style.cssText = 'position: fixed; top: 50px; right: 10px; padding: 10px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; z-index: 1000;';
    refreshButton.onclick = () => busAPI.checkForUpdates();
    document.body.appendChild(refreshButton);
});

// Listen for seating layout updates
window.addEventListener('seatingLayoutUpdated', function(event) {
    console.log('📊 Seating layout updated event received:', event.detail);
    
    // Calculate and display occupancy stats
    const layout = event.detail.layout;
    let totalSeats = 0;
    let occupiedSeats = 0;
    
    for (const row in layout) {
        for (const seat in layout[row]) {
            totalSeats++;
            if (layout[row][seat].class_id === 0) { // 0 = occupied
                occupiedSeats++;
            }
        }
    }
    
    const occupancyPercentage = totalSeats > 0 ? (occupiedSeats / totalSeats * 100).toFixed(1) : 0;
    
    console.log(`📈 Occupancy: ${occupiedSeats}/${totalSeats} (${occupancyPercentage}%)`);
    
    // Update occupancy display if element exists
    const occupancyElement = document.getElementById('occupancy-stats');
    if (occupancyElement) {
        occupancyElement.textContent = `Occupancy: ${occupiedSeats}/${totalSeats} (${occupancyPercentage}%)`;
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BusSeatingAPI;
} else {
    window.BusSeatingAPI = BusSeatingAPI;
}
