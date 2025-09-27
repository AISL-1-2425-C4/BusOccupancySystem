/**
 * Auto-refresh seating layout for newmorning.html and newafternoon.html
 * Add this script to your HTML files to enable real-time updates
 */

class SeatingAutoRefresh {
    constructor() {
        this.lastUpdated = null;
        this.pollInterval = 15000; // Check every 15 seconds
        this.isPolling = false;
        this.retryCount = 0;
        this.maxRetries = 3;
    }

    /**
     * Start auto-refresh polling
     */
    startAutoRefresh() {
        if (this.isPolling) {
            console.log('Auto-refresh already running');
            return;
        }

        console.log('🔄 Starting auto-refresh for seating layout');
        this.isPolling = true;
        
        // Initial check
        this.checkForUpdates();
        
        // Set up polling interval
        this.pollIntervalId = setInterval(() => {
            this.checkForUpdates();
        }, this.pollInterval);

        // Add visual indicator
        this.addRefreshIndicator();
    }

    /**
     * Stop auto-refresh polling
     */
    stopAutoRefresh() {
        if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
        }
        this.isPolling = false;
        this.removeRefreshIndicator();
        console.log('⏹️ Auto-refresh stopped');
    }

    /**
     * Check for seating layout updates
     */
    async checkForUpdates() {
        try {
            console.log('🔍 Checking for seating updates...');
            
            const response = await fetch('/row_seating_layout.json?' + Date.now()); // Cache busting
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const newData = await response.json();
            
            // Check if data has changed (simple comparison)
            const newDataString = JSON.stringify(newData);
            
            if (this.lastDataString && this.lastDataString !== newDataString) {
                console.log('🆕 New seating data detected, refreshing layout');
                this.refreshSeatingDisplay(newData);
                this.showUpdateNotification();
            } else if (!this.lastDataString) {
                console.log('📊 Initial seating data loaded');
                this.refreshSeatingDisplay(newData);
            } else {
                console.log('💤 No changes detected');
            }
            
            this.lastDataString = newDataString;
            this.retryCount = 0; // Reset retry count on success
            this.updateRefreshIndicator('success');
            
        } catch (error) {
            console.error('❌ Error checking for updates:', error);
            this.retryCount++;
            this.updateRefreshIndicator('error');
            
            if (this.retryCount >= this.maxRetries) {
                console.log('🔄 Max retries reached, continuing to poll...');
                this.retryCount = 0; // Reset for next cycle
            }
        }
    }

    /**
     * Refresh the seating display with new data
     */
    refreshSeatingDisplay(newData) {
        try {
            // Check if the page has a generateSeats function (common in seating layouts)
            if (typeof window.generateSeats === 'function') {
                window.generateSeats(newData);
            }
            // Check if there's a loadDynamicSeats function
            else if (typeof window.loadDynamicSeats === 'function') {
                window.seatLayoutData = newData;
                window.loadDynamicSeats();
            }
            // Check if there's an updateSeatingLayout function
            else if (typeof window.updateSeatingLayout === 'function') {
                window.updateSeatingLayout(newData);
            }
            // Generic approach - trigger a custom event
            else {
                window.dispatchEvent(new CustomEvent('seatingDataUpdated', {
                    detail: { data: newData }
                }));
            }
            
            // Update any occupancy statistics
            this.updateOccupancyStats(newData);
            
        } catch (error) {
            console.error('❌ Error refreshing seating display:', error);
        }
    }

    /**
     * Update occupancy statistics display
     */
    updateOccupancyStats(data) {
        try {
            let totalSeats = 0;
            let occupiedSeats = 0;
            
            // Count seats
            for (const row in data) {
                for (const seat in data[row]) {
                    totalSeats++;
                    if (data[row][seat].class_id === 0) { // 0 = occupied
                        occupiedSeats++;
                    }
                }
            }
            
            const occupancyPercentage = totalSeats > 0 ? (occupiedSeats / totalSeats * 100).toFixed(1) : 0;
            
            // Update occupancy display elements
            const occupancyElements = [
                document.getElementById('occupancy-stats'),
                document.getElementById('occupancy-percentage'),
                document.querySelector('.occupancy-info'),
                document.querySelector('.seat-stats')
            ];
            
            occupancyElements.forEach(element => {
                if (element) {
                    element.textContent = `${occupiedSeats}/${totalSeats} seats occupied (${occupancyPercentage}%)`;
                }
            });
            
            console.log(`📈 Occupancy: ${occupiedSeats}/${totalSeats} (${occupancyPercentage}%)`);
            
        } catch (error) {
            console.error('❌ Error updating occupancy stats:', error);
        }
    }

    /**
     * Show update notification to user
     */
    showUpdateNotification() {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10000;
            font-size: 14px;
            font-weight: 500;
            animation: slideIn 0.3s ease-out;
        `;
        notification.innerHTML = '🔄 Seating layout updated!';
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideIn 0.3s ease-out reverse';
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, 3000);
    }

    /**
     * Add refresh indicator to the page
     */
    addRefreshIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'refresh-indicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        indicator.innerHTML = `
            <div class="refresh-dot" style="width: 8px; height: 8px; border-radius: 50%; background: #4CAF50;"></div>
            <span>Auto-refresh active</span>
        `;
        document.body.appendChild(indicator);
    }

    /**
     * Update refresh indicator status
     */
    updateRefreshIndicator(status) {
        const indicator = document.getElementById('refresh-indicator');
        if (indicator) {
            const dot = indicator.querySelector('.refresh-dot');
            if (dot) {
                dot.style.background = status === 'success' ? '#4CAF50' : '#f44336';
            }
        }
    }

    /**
     * Remove refresh indicator
     */
    removeRefreshIndicator() {
        const indicator = document.getElementById('refresh-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Initialize auto-refresh when page loads
let seatingRefresh;

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing seating auto-refresh');
    
    seatingRefresh = new SeatingAutoRefresh();
    seatingRefresh.startAutoRefresh();
    
    // Add manual refresh button
    const refreshButton = document.createElement('button');
    refreshButton.textContent = '🔄 Refresh Now';
    refreshButton.style.cssText = `
        position: fixed;
        bottom: 70px;
        right: 20px;
        padding: 10px 15px;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 12px;
        z-index: 9999;
    `;
    refreshButton.onclick = () => {
        console.log('🔄 Manual refresh triggered');
        seatingRefresh.checkForUpdates();
    };
    document.body.appendChild(refreshButton);
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (seatingRefresh) {
        seatingRefresh.stopAutoRefresh();
    }
});

// Export for manual control
window.SeatingAutoRefresh = SeatingAutoRefresh;
