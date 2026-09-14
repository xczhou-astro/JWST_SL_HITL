class SLDetector {
    constructor() {
        this.hiddenImages = new Set(); // Images marked as non-SL (hidden)
        this.currentBatch = [];
        this.currentScores = {};
        this.isTraining = false;
        this.currentRound = 0;
        this.modelTrained = false;
        this.submitCooldown = false;
        this.isSubmitting = false;
        this.submitCooldownSeconds = 3;
        this.cooldownTimer = null;

        this.initializeElements();
        this.bindEvents();
        this.loadInitialData();
    }
    
    initializeElements() {
        // Selection controls
        this.submitSelectionsBtn = document.getElementById('submit-selections-btn');
        this.clearSelectionsBtn = document.getElementById('clear-selections-btn');
        this.resetModelBtn = document.getElementById('reset-model-btn');
        
        // Training overlay
        this.trainingOverlay = document.getElementById('training-overlay');
        this.trainingMessage = document.getElementById('training-message');

        // Display elements
        this.imageGrid = document.getElementById('image-grid');
        
        // Status elements
        this.currentRoundEl = document.getElementById('current-round');
        this.slCountEl = document.getElementById('sl-count');
        this.nonSlCountEl = document.getElementById('non-sl-count');
        this.testingSlCountEl = document.getElementById('testing-sl-count');
        this.testingNonSlCountEl = document.getElementById('testing-non-sl-count');
        this.recoveredCountEl = document.getElementById('recovered-count');
        this.availableCountEl = document.getElementById('available-count');
        this.totalSubmissionsEl = document.getElementById('total-submissions');
        
        // Visualization elements
        this.visualizationSection = document.getElementById('visualization-section');
        this.visualizationImage = document.getElementById('visualization-image');
        this.historyPlotsRow = document.getElementById('history-plots-row');
        this.selectionHistoryPanel = document.getElementById('selection-history-panel');
        this.selectionHistoryImage = document.getElementById('selection-history-image');
        this.medianRankHistoryPanel = document.getElementById('median-rank-history-panel');
        this.medianRankHistoryImage = document.getElementById('median-rank-history-image');
        this.historyImage = document.getElementById('history-image');
        
        // Notification
        this.notification = document.getElementById('notification');
        
        // Image popup modal elements
        this.imagePopupModal = document.getElementById('image-popup-modal');
        this.popupImage = document.getElementById('popup-image');
        this.popupInfo = document.getElementById('popup-info');
        this.popupZoomInfo = document.getElementById('popup-zoom-info');
        this.popupClose = document.getElementById('popup-close');
        this.imageZoomScale = 1.0;
        
        // Bind zoom handler so we can remove it later
        this.boundHandleImageZoom = this.handleImageZoom.bind(this);
    }
    
    bindEvents() {
        this.submitSelectionsBtn.addEventListener('click', () => this.submitSelections());
        this.clearSelectionsBtn.addEventListener('click', () => this.clearSelections());
        this.resetModelBtn.addEventListener('click', () => this.resetModel());
        
        // Image popup modal events
        this.popupClose.addEventListener('click', () => this.closeImagePopup());
        this.imagePopupModal.addEventListener('click', (e) => {
            if (e.target === this.imagePopupModal) {
                this.closeImagePopup();
            }
        });
        
        // Prevent right-click context menu on popup
        this.popupImage.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    async loadInitialData() {
        await this.updateStatus();
        await this.loadImages();
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/get_status');
            const data = await response.json();
            
            if (data.success) {
                this.updateCounts(data);
                this.modelTrained = data.model_trained;
                
                // Show visualization if model is trained
                if (this.modelTrained) {
                    this.showVisualization();
                }
            }
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }

    async loadImages() {
        try {
            console.log('Loading images...');
            this.showNotification('Loading images...', 'info');
            const response = await fetch('/api/get_images');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Received data:', data);
            
            if (data.error) {
                this.showNotification(data.error, 'error');
                return;
            }
            
            if (!data.success) {
                this.showNotification('Failed to load images', 'error');
                return;
            }
            
            if (!data.galaxy_names || data.galaxy_names.length === 0) {
                this.showNotification('No images available', 'warning');
                console.log('No images returned from server');
                return;
            }
            
            this.currentBatch = data.galaxy_names;
            this.currentScores = {};
            data.galaxy_names.forEach((name, index) => {
                this.currentScores[name] = data.scores[index];
            });
            
            console.log('Rendering', this.currentBatch.length, 'images');
            this.renderImageGrid();
            this.updateCounts(data);
            
            // Start submit cooldown
            this.startSubmitCooldown();
            
            this.showNotification(`Loaded ${this.currentBatch.length} images`, 'success');
        } catch (error) {
            this.showNotification('Error loading images', 'error');
            console.error('Error loading images:', error);
        }
    }

    renderImageGrid() {
        this.imageGrid.innerHTML = '';
        
        this.currentBatch.forEach((galaxyName, index) => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            imageItem.dataset.galaxyName = galaxyName;
            
            // Image content container
            const imageContent = document.createElement('div');
            imageContent.className = 'image-content';
            
            const img = document.createElement('img');
            img.src = `/images/${galaxyName}.jpg`;
            img.alt = `Galaxy ${galaxyName}`;
            img.loading = 'lazy';
            
            const overlay = document.createElement('div');
            overlay.className = 'image-overlay';
            overlay.textContent = galaxyName;
            
            const scoreOverlay = document.createElement('div');
            scoreOverlay.className = 'score-overlay';
            const score = this.currentScores[galaxyName] || 0.5;
            scoreOverlay.textContent = `Score: ${score.toFixed(3)}`;
            scoreOverlay.title = `Model confidence: ${score.toFixed(3)}`;
            
            imageContent.appendChild(img);
            imageContent.appendChild(overlay);
            imageContent.appendChild(scoreOverlay);
            
            // Add click event listener - left click to toggle hide/show
            imageItem.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleHidden(galaxyName, imageItem);
            });
            
            // Right-click to open image popup
            imageItem.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                this.openImagePopup(img.src, galaxyName, score);
            });
            
            imageItem.appendChild(imageContent);
            this.imageGrid.appendChild(imageItem);
        });
    }

    toggleHidden(galaxyName, imageItem) {
        if (this.hiddenImages.has(galaxyName)) {
            // Unhide the image
            this.hiddenImages.delete(galaxyName);
            imageItem.classList.remove('hidden');
        } else {
            // Hide the image (mark as non-SL)
            this.hiddenImages.add(galaxyName);
            imageItem.classList.add('hidden');
        }
    }
    
    async clearSelections() {
        // Show confirmation popup
        const confirmed = await this.showResetConfirmation();
        
        if (!confirmed) {
            return; // User cancelled
        }
        
        try {
            this.showNotification('Resetting round...', 'info');
            
            const response = await fetch('/app/reset_round', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Round reset successfully', 'success');
                
                // Update counts
                this.updateCounts(data);
                
                // Clear local selections
                this.hiddenImages.clear();
                
                // Load new images
                if (data.galaxy_names && data.scores) {
                    this.currentBatch = data.galaxy_names;
                    this.currentScores = {};
                    data.galaxy_names.forEach((name, index) => {
                        this.currentScores[name] = data.scores[index];
                    });
                    this.renderImageGrid();
                    this.startSubmitCooldown();
                } else {
                    await this.loadImages();
                }
                
                // Update model trained status
                this.modelTrained = data.model_trained;
                
                // Update visualization if needed
                if (this.modelTrained) {
                    this.showVisualization();
                } else {
                    this.visualizationSection.style.display = 'none';
                }
                
                // Update status
                await this.updateStatus();
            } else {
                this.showNotification(`Error resetting round: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            this.showNotification('Error resetting round', 'error');
            console.error('Error resetting round:', error);
        }
    }
    
    showResetConfirmation() {
        return new Promise((resolve) => {
            // Create modal overlay
            const modal = document.createElement('div');
            modal.className = 'confirmation-modal-overlay';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10000;
            `;
            
            // Create modal content
            const modalContent = document.createElement('div');
            modalContent.className = 'confirmation-modal-content';
            modalContent.style.cssText = `
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                width: 90%;
                text-align: center;
            `;
            
            // Create title
            const title = document.createElement('h2');
            title.textContent = 'Reset Round';
            title.style.cssText = 'margin-top: 0; margin-bottom: 15px; color: #333;';
            
            // Create message
            const message = document.createElement('p');
            message.textContent = 'Are you sure you want to reset the current round? This will reset all SL and non-SL selections to the previous round state.';
            message.style.cssText = 'margin-bottom: 25px; color: #666; line-height: 1.5;';
            
            // Create button container
            const buttonContainer = document.createElement('div');
            buttonContainer.style.cssText = 'display: flex; gap: 10px; justify-content: center;';
            
            // Create confirm button
            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = 'Confirm';
            confirmBtn.className = 'btn btn-primary';
            confirmBtn.style.cssText = 'padding: 10px 20px; cursor: pointer;';
            confirmBtn.onclick = () => {
                document.body.removeChild(modal);
                resolve(true);
            };
            
            // Create cancel button
            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'btn btn-secondary';
            cancelBtn.style.cssText = 'padding: 10px 20px; cursor: pointer;';
            cancelBtn.onclick = () => {
                document.body.removeChild(modal);
                resolve(false);
            };
            
            // Assemble modal
            buttonContainer.appendChild(confirmBtn);
            buttonContainer.appendChild(cancelBtn);
            modalContent.appendChild(title);
            modalContent.appendChild(message);
            modalContent.appendChild(buttonContainer);
            modal.appendChild(modalContent);
            
            // Add to page
            document.body.appendChild(modal);
            
            // Close on overlay click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    document.body.removeChild(modal);
                    resolve(false);
                }
            });
        });
    }
    
    showResetModelConfirmation() {
        return new Promise((resolve) => {
            // Create modal overlay
            const modal = document.createElement('div');
            modal.className = 'confirmation-modal-overlay';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10000;
            `;
            
            // Create modal content
            const modalContent = document.createElement('div');
            modalContent.className = 'confirmation-modal-content';
            modalContent.style.cssText = `
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                width: 90%;
                text-align: center;
            `;
            
            // Create title
            const title = document.createElement('h2');
            title.textContent = 'Reset Model';
            title.style.cssText = 'margin-top: 0; margin-bottom: 15px; color: #333;';
            
            // Create message
            const message = document.createElement('p');
            message.textContent = 'Are you sure you want to reset the model? This will reinitialize all ensembles to their initial state.';
            message.style.cssText = 'margin-bottom: 25px; color: #666; line-height: 1.5;';
            
            // Create button container
            const buttonContainer = document.createElement('div');
            buttonContainer.style.cssText = 'display: flex; gap: 10px; justify-content: center;';
            
            // Create confirm button
            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = 'Confirm';
            confirmBtn.className = 'btn btn-primary';
            confirmBtn.style.cssText = 'padding: 10px 20px; cursor: pointer;';
            confirmBtn.onclick = () => {
                document.body.removeChild(modal);
                resolve(true);
            };
            
            // Create cancel button
            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'btn btn-secondary';
            cancelBtn.style.cssText = 'padding: 10px 20px; cursor: pointer;';
            cancelBtn.onclick = () => {
                document.body.removeChild(modal);
                resolve(false);
            };
            
            // Assemble modal
            buttonContainer.appendChild(confirmBtn);
            buttonContainer.appendChild(cancelBtn);
            modalContent.appendChild(title);
            modalContent.appendChild(message);
            modalContent.appendChild(buttonContainer);
            modal.appendChild(modalContent);
            
            // Add to page
            document.body.appendChild(modal);
            
            // Close on overlay click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    document.body.removeChild(modal);
                    resolve(false);
                }
            });
        });
    }
    
    async resetModel() {
        // Show confirmation popup
        const confirmed = await this.showResetModelConfirmation();
        
        if (!confirmed) {
            return; // User cancelled
        }
        
        try {
            this.showNotification('Resetting model...', 'info');
            
            const response = await fetch('/app/reset_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Model reset successfully', 'success');
                
                // Update status
                await this.updateStatus();
            } else {
                this.showNotification(`Error resetting model: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            this.showNotification('Error resetting model', 'error');
            console.error('Error resetting model:', error);
        }
    }
    
    async submitSelections() {
        if (this.submitCooldown || this.isSubmitting || this.isTraining) {
            this.showNotification('Please wait before submitting again', 'warning');
            return;
        }

        // Get SL names (all visible images) and Non-SL names (hidden images)
        const slNames = this.currentBatch.filter(name => !this.hiddenImages.has(name));
        const nonSlNames = Array.from(this.hiddenImages);

        if (slNames.length === 0 && nonSlNames.length === 0) {
            this.showNotification('No images to submit', 'error');
            return;
        }

        this.isSubmitting = true;
        this.submitSelectionsBtn.disabled = true;
        this.submitSelectionsBtn.textContent = 'Submitting...';

        try {
            const response = await fetch('/app/submit_selections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sl_names: slNames,
                    non_sl_names: nonSlNames
                })
            });

            if (response.status === 429) {
                this.showNotification('Submit already in progress', 'warning');
                return;
            }

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.showNotification('Selections submitted successfully', 'success');

                this.updateCounts(data);
                this.hiddenImages.clear();

                if (data.should_train) {
                    this.showNotification('100 selections reached! Starting automatic training...', 'info');
                    await this.autoTrain();
                } else if (data.galaxy_names && data.scores) {
                    this.currentBatch = data.galaxy_names;
                    this.currentScores = {};
                    data.galaxy_names.forEach((name, index) => {
                        this.currentScores[name] = data.scores[index];
                    });
                    this.renderImageGrid();
                    this.startSubmitCooldown();
                    this.showNotification(`Loaded ${this.currentBatch.length} new images`, 'success');
                } else {
                    await this.loadImages();
                }
            } else {
                this.showNotification(`Error submitting selections: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            this.showNotification('Error submitting selections', 'error');
            console.error('Error submitting selections:', error);
        } finally {
            if (this.isSubmitting) {
                this.isSubmitting = false;
                if (!this.submitCooldown && !this.isTraining) {
                    this.submitSelectionsBtn.disabled = false;
                    this.submitSelectionsBtn.textContent = 'Submit';
                }
            }
        }
    }
    
    showTrainingOverlay(message = 'Please wait while the model is being trained.') {
        this.trainingOverlay.style.display = 'flex';
        this.trainingMessage.textContent = message;
    }
    
    hideTrainingOverlay() {
        this.trainingOverlay.style.display = 'none';
    }
    
    playNotificationSound() {
        try {
            // Create audio context (handle browser compatibility)
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) {
                console.warn('Web Audio API not supported');
                return;
            }
            
            const audioContext = new AudioContext();
            
            // Create oscillator for a pleasant notification sound
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            // Connect nodes
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            // Configure sound: two-tone chime (success sound)
            const now = audioContext.currentTime;
            oscillator.frequency.setValueAtTime(800, now);
            oscillator.frequency.setValueAtTime(1000, now + 0.1);
            oscillator.type = 'sine';
            
            // Set volume envelope (fade in/out smoothly)
            gainNode.gain.setValueAtTime(0, now);
            gainNode.gain.linearRampToValueAtTime(0.3, now + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
            
            // Play sound
            oscillator.start(now);
            oscillator.stop(now + 0.5);
            
            // Clean up audio context after sound finishes
            oscillator.onended = () => {
                audioContext.close().catch(() => {});
            };
        } catch (error) {
            console.warn('Could not play notification sound:', error);
        }
    }
    
    async waitForScoringComplete(expectedJobSeq) {
        const maxMs = 60 * 60 * 1000;
        const t0 = Date.now();
        const want = Number(expectedJobSeq);
        const useJobHandshake = Number.isFinite(want) && want > 0;
        while (Date.now() - t0 < maxMs) {
            const response = await fetch(`/api/get_status?t=${Date.now()}`, { cache: 'no-store' });
            const data = await response.json();
            if (data.scoring_last_error) {
                throw new Error(data.scoring_last_error);
            }
            if (useJobHandshake) {
                const done = Number(data.scoring_completed_seq) || 0;
                if (data.success && done >= want) {
                    return;
                }
            } else if (data.success && !data.scoring_in_progress) {
                return;
            }
            await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        throw new Error('Scoring is taking too long. Check the server log and try refreshing the page.');
    }

    async autoTrain() {
        this.isTraining = true;
        this.submitSelectionsBtn.disabled = true;
        this.showTrainingOverlay('Training model automatically...');
        
        const epochs = 300;
        
        try {
            const response = await fetch('/api/run_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ epochs })
            });

            if (response.status === 429) {
                this.showNotification('Training already in progress', 'warning');
                return;
            }

            const raw = await response.text();
            let data;
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch (parseErr) {
                console.error('Training response is not valid JSON:', raw.slice(0, 500));
                throw new Error('Server returned non-JSON (often invalid Infinity/NaN in JSON). Check app logs.');
            }
            
            if (data.success) {
                if (data.skipped) {
                    this.showNotification('Round 0 complete. Continue labeling with next batch.', 'info');
                    await this.updateStatus();
                    await this.loadImages();
                    return;
                }
                this.modelTrained = true;

                if (data.scoring_async) {
                    this.showTrainingOverlay('Scoring full catalog (may take several minutes; safe to leave this tab open)...');
                    await this.waitForScoringComplete(data.scoring_job_seq);
                }

                this.playNotificationSound();
                this.showNotification('Model trained successfully!', 'success');

                this.showVisualization();
                await this.updateStatus();
                await this.loadImages();
            } else {
                this.showNotification('Training failed: ' + (data.error || `HTTP ${response.status}` || 'Unknown error'), 'error');
            }
        } catch (error) {
            this.showNotification(error.message || 'Error during training', 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            this.hideTrainingOverlay();
        }
    }
    
    loadOptionalHistoryImage(imgEl, panelEl, url) {
        if (!imgEl || !panelEl) {
            return;
        }
        panelEl.style.display = 'none';
        imgEl.onerror = () => {
            panelEl.style.display = 'none';
            this.updateHistoryPlotsRowVisibility();
        };
        imgEl.onload = () => {
            panelEl.style.display = 'block';
            this.updateHistoryPlotsRowVisibility();
        };
        imgEl.src = url;
    }

    updateHistoryPlotsRowVisibility() {
        if (!this.historyPlotsRow) {
            return;
        }
        const selectionVisible = this.selectionHistoryPanel
            && this.selectionHistoryPanel.style.display !== 'none';
        const medianVisible = this.medianRankHistoryPanel
            && this.medianRankHistoryPanel.style.display !== 'none';
        this.historyPlotsRow.style.display = (selectionVisible || medianVisible)
            ? 'grid'
            : 'none';
    }

    showVisualization() {
        try {
            if (this.modelTrained) {
                this.visualizationSection.style.display = 'block';
                const cacheBuster = new Date().getTime();
                this.visualizationImage.src = `/visualizations/visualizations.png?t=${cacheBuster}`;
                this.historyImage.src = `/visualizations/history.png?t=${cacheBuster}`;
                if (this.historyPlotsRow) {
                    this.historyPlotsRow.style.display = 'none';
                }
                this.loadOptionalHistoryImage(
                    this.selectionHistoryImage,
                    this.selectionHistoryPanel,
                    `/visualizations/selection_history.png?t=${cacheBuster}`,
                );
                this.loadOptionalHistoryImage(
                    this.medianRankHistoryImage,
                    this.medianRankHistoryPanel,
                    `/visualizations/median_rank_history.png?t=${cacheBuster}`,
                );
            } else {
                console.log('No visualization available yet');
            }
        } catch (error) {
            console.error('Error loading visualization:', error);
        }
    }
    
    
    updateCounts(data) {
        this.currentRoundEl.textContent = data.round || this.currentRound;
        this.slCountEl.textContent = data.sl_count || 0;
        this.nonSlCountEl.textContent = data.non_sl_count || 0;
        this.testingSlCountEl.textContent = data.testing_sl_count ?? this.testingSlCountEl.textContent ?? 0;
        this.testingNonSlCountEl.textContent = data.testing_non_sl_count ?? this.testingNonSlCountEl.textContent ?? 0;
        this.recoveredCountEl.textContent = data.recovered_count ?? data.injected_recovered_count ?? this.recoveredCountEl.textContent ?? 0;
        this.availableCountEl.textContent = data.available_count || 0;
        this.totalSubmissionsEl.textContent = data.total_submissions || 0;
    }
    
    startSubmitCooldown() {
        // Clear any existing timer first
        if (this.cooldownTimer) {
            clearInterval(this.cooldownTimer);
            this.cooldownTimer = null;
        }

        this.isSubmitting = false;
        this.submitCooldown = true;
        this.submitSelectionsBtn.disabled = true;

        let countdown = this.submitCooldownSeconds;
        this.submitSelectionsBtn.textContent = `Submit (${countdown}s)`;
        
        this.cooldownTimer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                this.submitSelectionsBtn.textContent = `Submit (${countdown}s)`;
            } else {
                clearInterval(this.cooldownTimer);
                this.cooldownTimer = null;
                this.submitCooldown = false;
                this.submitSelectionsBtn.disabled = false;
                this.submitSelectionsBtn.textContent = 'Submit';
            }
        }, 1000);
    }
    
    showNotification(message, type = 'info') {
        this.notification.textContent = message;
        this.notification.className = `notification ${type}`;
        this.notification.classList.add('show');
        
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, 3000);
    }
    
    openImagePopup(imageSrc, galaxyName, score) {
        this.popupImage.src = imageSrc;
        this.popupInfo.textContent = `${galaxyName} | Score: ${score.toFixed(3)}`;
        this.imageZoomScale = 1.0;
        this.popupImage.style.transform = `scale(${this.imageZoomScale})`;
        this.imagePopupModal.classList.add('active');
        this.updateZoomInfo();
        
        // Add scroll event listener for zooming
        this.imagePopupModal.addEventListener('wheel', this.boundHandleImageZoom, { passive: false });
        
        // Prevent body scroll when modal is open
        document.body.style.overflow = 'hidden';
    }
    
    closeImagePopup() {
        this.imagePopupModal.classList.remove('active');
        this.imageZoomScale = 1.0;
        this.popupImage.style.transform = 'scale(1)';
        
        // Remove scroll event listener
        this.imagePopupModal.removeEventListener('wheel', this.boundHandleImageZoom);
        
        // Restore body scroll
        document.body.style.overflow = '';
    }
    
    handleImageZoom(e) {
        e.preventDefault();
        
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        this.imageZoomScale = Math.max(0.5, Math.min(5.0, this.imageZoomScale + delta));
        
        this.popupImage.style.transform = `scale(${this.imageZoomScale})`;
        this.updateZoomInfo();
    }
    
    updateZoomInfo() {
        const zoomPercent = Math.round(this.imageZoomScale * 100);
        this.popupZoomInfo.textContent = `Zoom: ${zoomPercent}% | Scroll to zoom`;
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new SLDetector();
});
