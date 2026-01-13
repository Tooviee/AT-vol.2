// Main JavaScript for Trading Dashboard

// Auto-refresh functionality
let autoRefreshInterval = null;

function startAutoRefresh(interval = 30000) {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        // Reload current page
        if (window.location.pathname === '/') {
            window.location.reload();
        }
    }, interval);
}

// Stop auto-refresh
function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// Format currency
function formatCurrency(amount, currency = 'KRW') {
    if (currency === 'KRW') {
        return new Intl.NumberFormat('ko-KR').format(Math.round(amount)) + ' KRW';
    } else {
        return '$' + amount.toFixed(2);
    }
}

// Format number with commas
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Start auto-refresh on dashboard (every 30 seconds)
    if (window.location.pathname === '/') {
        startAutoRefresh(30000);
    }
});
