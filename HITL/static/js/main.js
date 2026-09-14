let currentBatch = [];
let selectedSources = new Set();
let labeledCount = 0;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    updateStatus();
    getNewBatch();
});

// Get new batch of images
async function getNewBatch() {
    try {
        showLoading(true);
        const response = await fetch('/get_batch');
        const data = await response.json();
        
        currentBatch = data.batch;
        selectedSources.clear();
        
        displayImages(currentBatch);
        updateSubmitButton();
        
    } catch (error) {
        console.error('Error getting new batch:', error);
        alert('Error loading new batch. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Display images in grid
function displayImages(batch) {
    const grid = document.getElementById('image-grid');
    grid.innerHTML = '';
    
    if (batch.length === 0) {
        grid.innerHTML = '<div class="col-12 text-center"><h4>No more sources available for labeling</h4></div>';
        return;
    }
    
    batch.forEach(sourceName => {
        const imageItem = document.createElement('div');
        imageItem.className = 'image-item';
        imageItem.dataset.source = sourceName;
        
        imageItem.innerHTML = `
            <img src="/image/${sourceName}" alt="${sourceName}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='">
            <div class="image-label">${sourceName}</div>
        `;
        
        // Add click event listeners
        imageItem.addEventListener('click', function(e) {
            e.preventDefault();
            toggleSelection(sourceName);
        });
        
        imageItem.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            deselectSource(sourceName);
        });
        
        grid.appendChild(imageItem);
    });
}

// Toggle source selection
function toggleSelection(sourceName) {
    const imageItem = document.querySelector(`[data-source="${sourceName}"]`);
    
    if (selectedSources.has(sourceName)) {
        selectedSources.delete(sourceName);
        imageItem.classList.remove('selected');
    } else {
        selectedSources.add(sourceName);
        imageItem.classList.add('selected');
    }
    
    updateSubmitButton();
}

// Deselect source
function deselectSource(sourceName) {
    const imageItem = document.querySelector(`[data-source="${sourceName}"]`);
    
    if (selectedSources.has(sourceName)) {
        selectedSources.delete(sourceName);
        imageItem.classList.remove('selected');
        updateSubmitButton();
    }
}

// Update submit button state
function updateSubmitButton() {
    const submitBtn = document.getElementById('submit-btn');
    const hasSelections = selectedSources.size > 0 || currentBatch.length > 0;
    
    submitBtn.disabled = !hasSelections;
    submitBtn.textContent = `Submit Labels (${selectedSources.size}/${currentBatch.length})`;
}

// Submit labels
async function submitLabels() {
    if (currentBatch.length === 0) {
        alert('No batch to submit');
        return;
    }
    
    try {
        showLoading(true);
        
        // Create labels object: 1 for selected (interested), 0 for not selected
        const labels = {};
        currentBatch.forEach(sourceName => {
            labels[sourceName] = selectedSources.has(sourceName) ? 1 : 0;
        });
        
        const response = await fetch('/submit_labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ labels: labels })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            labeledCount = data.total_labeled;
            updateStatus();
            
            // Show success message
            showMessage(`Successfully submitted labels for ${currentBatch.length} sources!`, 'success');
            
            // Get new batch automatically
            setTimeout(() => {
                getNewBatch();
            }, 1000);
        } else {
            throw new Error('Failed to submit labels');
        }
        
    } catch (error) {
        console.error('Error submitting labels:', error);
        alert('Error submitting labels. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Train model
async function trainModel() {
    const epochs = parseInt(document.getElementById('epochs').value);
    
    if (labeledCount < 10) {
        alert('Need at least 10 labeled samples to train the model');
        return;
    }
    
    try {
        showLoading(true);
        document.getElementById('training-status').innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div>Training...';
        
        const response = await fetch('/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ epochs: epochs })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('training-status').innerHTML = '<div class="text-success">✓ ' + data.message + '</div>';
            updateStatus();
            showMessage('Model trained successfully!', 'success');
        } else {
            document.getElementById('training-status').innerHTML = '<div class="text-danger">✗ ' + data.message + '</div>';
            showMessage('Training failed: ' + data.message, 'error');
        }
        
    } catch (error) {
        console.error('Error training model:', error);
        document.getElementById('training-status').innerHTML = '<div class="text-danger">✗ Training failed</div>';
        alert('Error training model. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Evaluate model
async function evaluateModel() {
    try {
        showLoading(true);
        
        const response = await fetch('/evaluate');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('threshold-100').textContent = data.threshold_100.toFixed(4);
            document.getElementById('threshold').value = data.threshold_100.toFixed(4);
            document.getElementById('threshold-controls').style.display = 'block';
            
            updateStatus();
            showMessage('Model evaluation completed!', 'success');
            
            // Load histogram
            loadHistogram();
            
        } else {
            showMessage('Evaluation failed: ' + data.message, 'error');
        }
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        alert('Error evaluating model. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Analyze threshold
async function analyzeThreshold() {
    const threshold = parseFloat(document.getElementById('threshold').value);
    
    try {
        const response = await fetch('/analyze_threshold', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ threshold: threshold })
        });
        
        const data = await response.json();
        console.log('Threshold analysis response:', data);
        
        if (data && !data.error) {
            const resultsDiv = document.getElementById('threshold-results');
            resultsDiv.innerHTML = `
                <h6>Threshold Analysis (${threshold.toFixed(4)})</h6>
                <hr>
                <div class="row">
                    <div class="col-6">
                        <strong>Working Sources:</strong><br>
                        Total: ${data.working_total}<br>
                        Dropped: ${data.working_dropped} (${(data.working_drop_rate * 100).toFixed(1)}%)<br>
                        Kept: ${data.working_kept}
                    </div>
                    <div class="col-6">
                        <strong>Holdout Sources:</strong><br>
                        Total: ${data.holdout_total}<br>
                        Dropped: ${data.holdout_dropped} (${(data.holdout_drop_rate * 100).toFixed(1)}%)<br>
                        Kept: ${data.holdout_kept}
                    </div>
                </div>
            `;
            
            // Enable drop button after successful analysis
            document.getElementById('drop-btn').disabled = false;
        } else if (data && data.error) {
            const resultsDiv = document.getElementById('threshold-results');
            resultsDiv.innerHTML = `<div class="text-warning">${data.error}</div>`;
            document.getElementById('drop-btn').disabled = true;
        } else {
            console.error('No data received from threshold analysis');
            const resultsDiv = document.getElementById('threshold-results');
            resultsDiv.innerHTML = `<div class="text-danger">No analysis data received</div>`;
            document.getElementById('drop-btn').disabled = true;
        }
        
    } catch (error) {
        console.error('Error analyzing threshold:', error);
        alert('Error analyzing threshold. Please try again.');
    }
}

// Load histogram
async function loadHistogram() {
    try {
        const response = await fetch('/get_histogram');
        const data = await response.json();
        
        if (data.histogram) {
            document.getElementById('histogram-img').src = 'data:image/png;base64,' + data.histogram;
            document.getElementById('histogram-container').style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error loading histogram:', error);
    }
}

// Update status display
async function updateStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        document.getElementById('labeled-count').textContent = data.labeled_count;
        document.getElementById('positive-count').textContent = data.positive_labels;
        document.getElementById('negative-count').textContent = data.negative_labels;
        document.getElementById('dropped-count').textContent = data.dropped_count;
        document.getElementById('remaining-count').textContent = data.remaining_count;
        document.getElementById('model-status').textContent = data.model_trained ? 'Trained' : 'Not Trained';
        document.getElementById('eval-status').textContent = data.scores_available ? 'Available' : 'Not Available';
        
        labeledCount = data.labeled_count;
        
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Show loading state
function showLoading(loading) {
    const elements = document.querySelectorAll('button, .image-item');
    elements.forEach(el => {
        if (loading) {
            el.classList.add('loading');
        } else {
            el.classList.remove('loading');
        }
    });
}

// Show message
function showMessage(message, type) {
    const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.querySelector('.main-content').insertBefore(alertDiv, document.querySelector('.main-content').firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Drop sources with confirmation
async function dropSources() {
    const threshold = parseFloat(document.getElementById('threshold').value);
    
    if (!confirm(`Are you sure you want to drop all sources with scores below ${threshold.toFixed(4)}? This action cannot be undone.`)) {
        return;
    }
    
    try {
        showLoading(true);
        
        const response = await fetch('/drop_sources', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ threshold: threshold })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage(data.message, 'success');
            updateStatus();
            // Hide threshold controls after successful drop to prevent re-dropping
            document.getElementById('threshold-controls').style.display = 'none';
            showMessage('Threshold controls hidden. Re-evaluate model to continue filtering.', 'info');
        } else {
            showMessage('Drop failed: ' + data.message, 'error');
        }
        
    } catch (error) {
        console.error('Error dropping sources:', error);
        alert('Error dropping sources. Please try again.');
    } finally {
        showLoading(false);
    }
}


// Auto-update threshold analysis when threshold input changes
document.addEventListener('DOMContentLoaded', function() {
    const thresholdInput = document.getElementById('threshold');
    if (thresholdInput) {
        thresholdInput.addEventListener('input', function() {
            if (document.getElementById('threshold-controls').style.display !== 'none') {
                analyzeThreshold();
                // Disable drop button when threshold changes
                document.getElementById('drop-btn').disabled = true;
            }
        });
    }
});
