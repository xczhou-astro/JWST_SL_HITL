class GalleryManager {
    constructor() {
        this.slItems = [];
        this.nonSlItems = [];
        this.testingSlItems = [];
        this.testingNonSlItems = [];
        this.currentSLPage = 1;
        this.currentNonSLPage = 1;
        this.currentTestingSLPage = 1;
        this.currentTestingNonSLPage = 1;
        this.itemsPerPage = 10;
        this.slTotalPages = 1;
        this.nonSlTotalPages = 1;
        this.testingSlTotalPages = 1;
        this.testingNonSlTotalPages = 1;
        this.currentGallery = 'sl'; // 'sl' or 'non_sl' or 'testing_sl' or 'testing_non_sl'
        
        this.initializeElements();
        this.loadGalleryData();
    }
    
    initializeElements() {
        // SL Gallery elements
        this.slGalleryGrid = document.getElementById('sl-gallery-grid');
        this.slCurrentPageEl = document.getElementById('sl-current-page');
        this.slTotalPagesEl = document.getElementById('sl-total-pages');
        this.slTotalCountEl = document.getElementById('sl-total-count');
        this.slConfirmedCountEl = document.getElementById('sl-confirmed-count');
        this.slSelectedCountEl = document.getElementById('sl-selected-count');
        
        // Non-SL Gallery elements
        this.nonSlGalleryGrid = document.getElementById('non-sl-gallery-grid');
        this.nonSlCurrentPageEl = document.getElementById('non-sl-current-page');
        this.nonSlTotalPagesEl = document.getElementById('non-sl-total-pages');
        this.nonSlTotalCountEl = document.getElementById('non-sl-total-count');

        // Testing SL Gallery elements
        this.testingSlGalleryGrid = document.getElementById('testing-sl-gallery-grid');
        this.testingSlCurrentPageEl = document.getElementById('testing-sl-current-page');
        this.testingSlTotalPagesEl = document.getElementById('testing-sl-total-pages');
        this.testingSlTotalCountEl = document.getElementById('testing-sl-total-count');
        this.testingSlRecoveredCountEl = document.getElementById('testing-sl-recovered-count');

        // Testing Non-SL Gallery elements
        this.testingNonSlGalleryGrid = document.getElementById('testing-non-sl-gallery-grid');
        this.testingNonSlCurrentPageEl = document.getElementById('testing-non-sl-current-page');
        this.testingNonSlTotalPagesEl = document.getElementById('testing-non-sl-total-pages');
        this.testingNonSlTotalCountEl = document.getElementById('testing-non-sl-total-count');
        
        // Sections
        this.slSection = document.getElementById('sl-gallery-section');
        this.nonSlSection = document.getElementById('non-sl-gallery-section');
        this.testingSlSection = document.getElementById('testing-sl-gallery-section');
        this.testingNonSlSection = document.getElementById('testing-non-sl-gallery-section');
        
        // Tab buttons
        this.slTabBtn = document.getElementById('sl-tab-btn');
        this.nonSlTabBtn = document.getElementById('non-sl-tab-btn');
        this.testingSlTabBtn = document.getElementById('testing-sl-tab-btn');
        this.testingNonSlTabBtn = document.getElementById('testing-non-sl-tab-btn');
        
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
        
        // Bind popup events
        this.popupClose.addEventListener('click', () => this.closeImagePopup());
        this.imagePopupModal.addEventListener('click', (e) => {
            if (e.target === this.imagePopupModal) {
                this.closeImagePopup();
            }
        });
        this.popupImage.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    async loadGalleryData() {
        try {
            this.showNotification('Loading gallery...', 'info');
            
            const response = await fetch('/api/get_gallery_data');
            const data = await response.json();
            
            console.log('Gallery data received:', data);
            
            if (data.success) {
                console.log('Total items:', data.items.length);
                
                // Separate SL and non-SL items
                this.slItems = data.items.filter(item => item.type === 'sl');
                this.nonSlItems = data.items.filter(item => item.type === 'non_sl');
                this.testingSlItems = data.testing_sl_items || [];
                this.testingNonSlItems = data.testing_non_sl_items || [];
                
                console.log('SL items:', this.slItems.length);
                console.log('Non-SL items:', this.nonSlItems.length);
                if (this.nonSlItems.length > 0) {
                    console.log('First non-SL item:', this.nonSlItems[0]);
                }
                
                this.slTotalPages = Math.ceil(this.slItems.length / this.itemsPerPage) || 1;
                this.nonSlTotalPages = Math.ceil(this.nonSlItems.length / this.itemsPerPage) || 1;
                this.testingSlTotalPages = Math.ceil(this.testingSlItems.length / this.itemsPerPage) || 1;
                this.testingNonSlTotalPages = Math.ceil(this.testingNonSlItems.length / this.itemsPerPage) || 1;
                
                // Update SL counts
                this.slTotalCountEl.textContent = this.slItems.length;
                this.slConfirmedCountEl.textContent = data.confirmed_count;
                this.slSelectedCountEl.textContent = data.selected_sl_count;
                
                // Update Non-SL counts
                this.nonSlTotalCountEl.textContent = this.nonSlItems.length;

                // Update Testing SL counts
                this.testingSlTotalCountEl.textContent = data.testing_sl_count || this.testingSlItems.length;
                this.testingSlRecoveredCountEl.textContent = data.testing_sl_recovered_count || 0;
                this.testingNonSlTotalCountEl.textContent = data.testing_non_sl_count || this.testingNonSlItems.length;
                
                // Render first page of SL gallery
                this.renderSLPage(1);
                this.showNotification(`Loaded ${this.slItems.length} SL and ${this.nonSlItems.length} non-SL images`, 'success');
            } else {
                this.showNotification('Error loading gallery data', 'error');
            }
        } catch (error) {
            console.error('Error loading gallery:', error);
            this.showNotification('Error loading gallery', 'error');
        }
    }

    renderTestingSLPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.testingSlTotalPages) {
            return;
        }

        this.currentTestingSLPage = pageNumber;
        this.testingSlCurrentPageEl.textContent = this.currentTestingSLPage;
        this.testingSlTotalPagesEl.textContent = this.testingSlTotalPages;

        const startIndex = (this.currentTestingSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.testingSlItems.length);
        const pageItems = this.testingSlItems.slice(startIndex, endIndex);

        this.testingSlGalleryGrid.innerHTML = '';

        if (pageItems.length === 0) {
            this.testingSlGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No Testing SL Images</h3>
                    <p>Testing SL pool is currently empty.</p>
                </div>
            `;
        } else {
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.testingSlGalleryGrid);
            });
        }
    }

    renderTestingNonSLPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.testingNonSlTotalPages) {
            return;
        }

        this.currentTestingNonSLPage = pageNumber;
        this.testingNonSlCurrentPageEl.textContent = this.currentTestingNonSLPage;
        this.testingNonSlTotalPagesEl.textContent = this.testingNonSlTotalPages;

        const startIndex = (this.currentTestingNonSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.testingNonSlItems.length);
        const pageItems = this.testingNonSlItems.slice(startIndex, endIndex);

        this.testingNonSlGalleryGrid.innerHTML = '';

        if (pageItems.length === 0) {
            this.testingNonSlGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No Testing Non-SL Images</h3>
                    <p>Testing non-SL pool is currently empty.</p>
                </div>
            `;
        } else {
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.testingNonSlGalleryGrid);
            });
        }
    }
    
    renderSLPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.slTotalPages) {
            return;
        }
        
        this.currentSLPage = pageNumber;
        this.slCurrentPageEl.textContent = this.currentSLPage;
        this.slTotalPagesEl.textContent = this.slTotalPages;
        
        const startIndex = (this.currentSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.slItems.length);
        const pageItems = this.slItems.slice(startIndex, endIndex);
        
        this.slGalleryGrid.innerHTML = '';
        
        if (pageItems.length === 0) {
            this.slGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No SL Images Yet</h3>
                    <p>Start selecting strong lensing candidates!</p>
                </div>
            `;
        } else {
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.slGalleryGrid);
            });
        }
    }
    
    renderNonSLPage(pageNumber) {
        console.log('renderNonSLPage called with page:', pageNumber);
        console.log('Total non-SL items:', this.nonSlItems.length);
        console.log('Total pages:', this.nonSlTotalPages);
        
        if (pageNumber < 1 || pageNumber > this.nonSlTotalPages) {
            console.log('Page number out of range');
            return;
        }
        
        this.currentNonSLPage = pageNumber;
        this.nonSlCurrentPageEl.textContent = this.currentNonSLPage;
        this.nonSlTotalPagesEl.textContent = this.nonSlTotalPages;
        
        const startIndex = (this.currentNonSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.nonSlItems.length);
        const pageItems = this.nonSlItems.slice(startIndex, endIndex);
        
        console.log('Rendering items from', startIndex, 'to', endIndex);
        console.log('Page items:', pageItems.length);
        
        this.nonSlGalleryGrid.innerHTML = '';
        
        if (pageItems.length === 0) {
            console.log('No items to render - showing empty message');
            this.nonSlGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No Non-SL Images Yet</h3>
                    <p>Start marking non-SL images!</p>
                </div>
            `;
        } else {
            console.log('Rendering', pageItems.length, 'items');
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.nonSlGalleryGrid);
            });
        }
    }
    
    renderGalleryItem(item, gridElement) {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        
        // Change border color for different item types
        if (item.type === 'non_sl') {
            galleryItem.style.borderColor = '#e67e22';
        } else if (item.type === 'testing_sl') {
            galleryItem.style.borderColor = '#8e44ad';
        } else if (item.type === 'testing_non_sl') {
            galleryItem.style.borderColor = '#d35400';
        }
        
        const img = document.createElement('img');
        img.src = `/images/${item.name}.jpg`;
        img.alt = `Galaxy ${item.name}`;
        img.loading = 'lazy';
        
        const overlay = document.createElement('div');
        overlay.className = 'gallery-item-overlay';
        overlay.textContent = item.name;
        
        // Right-click to open image popup
        galleryItem.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            const infoText = item.grade && item.grade !== 'N/A' 
                ? `${item.name} | Grade: ${item.grade}` 
                : item.name;
            this.openImagePopup(img.src, infoText);
        });
        
        galleryItem.appendChild(img);
        galleryItem.appendChild(overlay);
        
        if (item.is_recovered) {
            const recoveredBadge = document.createElement('div');
            recoveredBadge.className = 'recovered-badge';
            recoveredBadge.textContent = 'RECOVERED';
            galleryItem.appendChild(recoveredBadge);
        } else if (item.is_confirmed) {
            const confirmedBadge = document.createElement('div');
            confirmedBadge.className = 'confirmed-badge';
            confirmedBadge.textContent = 'COWLS';
            galleryItem.appendChild(confirmedBadge);
        } else if (item.is_selected_sl) {
            const selectedBadge = document.createElement('div');
            selectedBadge.className = 'confirmed-badge';
            selectedBadge.style.background = 'rgba(46, 204, 113, 0.9)';
            selectedBadge.textContent = 'SELECTED SL';
            galleryItem.appendChild(selectedBadge);
        } else if (item.is_selected_non_sl) {
            const selectedBadge = document.createElement('div');
            selectedBadge.className = 'confirmed-badge';
            selectedBadge.style.background = 'rgba(230, 126, 34, 0.9)';
            selectedBadge.textContent = 'SELECTED NON-SL';
            galleryItem.appendChild(selectedBadge);
        }
        
        if (item.grade && item.grade !== 'N/A') {
            const gradeBadge = document.createElement('div');
            gradeBadge.className = 'gallery-item-grade';
            gradeBadge.textContent = `Grade: ${item.grade}`;
            galleryItem.appendChild(gradeBadge);
        }
        
        gridElement.appendChild(galleryItem);
    }
    
    showNotification(message, type = 'info') {
        this.notification.textContent = message;
        this.notification.className = `notification ${type}`;
        this.notification.classList.add('show');
        
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, 3000);
    }
    
    openImagePopup(imageSrc, infoText) {
        this.popupImage.src = imageSrc;
        this.popupInfo.textContent = infoText;
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

// Global functions for gallery switching and pagination
let galleryManager;

function showSLGallery() {
    if (galleryManager) {
        galleryManager.slSection.style.display = 'block';
        galleryManager.nonSlSection.style.display = 'none';
        galleryManager.testingSlSection.style.display = 'none';
        galleryManager.testingNonSlSection.style.display = 'none';
        galleryManager.slTabBtn.className = 'btn btn-primary';
        galleryManager.nonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingNonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.currentGallery = 'sl';
    }
}

function showNonSLGallery() {
    console.log('showNonSLGallery called');
    if (galleryManager) {
        console.log('Gallery manager exists');
        galleryManager.slSection.style.display = 'none';
        galleryManager.nonSlSection.style.display = 'block';
        galleryManager.testingSlSection.style.display = 'none';
        galleryManager.testingNonSlSection.style.display = 'none';
        galleryManager.slTabBtn.className = 'btn btn-secondary';
        galleryManager.nonSlTabBtn.className = 'btn btn-primary';
        galleryManager.testingSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingNonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.currentGallery = 'non_sl';
        console.log('Calling renderNonSLPage(1)');
        // Always render when switching to non-SL tab
        galleryManager.renderNonSLPage(1);
    } else {
        console.log('Gallery manager does not exist!');
    }
}

function showTestingSLGallery() {
    if (galleryManager) {
        galleryManager.slSection.style.display = 'none';
        galleryManager.nonSlSection.style.display = 'none';
        galleryManager.testingSlSection.style.display = 'block';
        galleryManager.testingNonSlSection.style.display = 'none';
        galleryManager.slTabBtn.className = 'btn btn-secondary';
        galleryManager.nonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingSlTabBtn.className = 'btn btn-primary';
        galleryManager.testingNonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.currentGallery = 'testing_sl';
        galleryManager.renderTestingSLPage(1);
    }
}

function showTestingNonSLGallery() {
    if (galleryManager) {
        galleryManager.slSection.style.display = 'none';
        galleryManager.nonSlSection.style.display = 'none';
        galleryManager.testingSlSection.style.display = 'none';
        galleryManager.testingNonSlSection.style.display = 'block';
        galleryManager.slTabBtn.className = 'btn btn-secondary';
        galleryManager.nonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingSlTabBtn.className = 'btn btn-secondary';
        galleryManager.testingNonSlTabBtn.className = 'btn btn-primary';
        galleryManager.currentGallery = 'testing_non_sl';
        galleryManager.renderTestingNonSLPage(1);
    }
}

// SL Gallery pagination
function nextSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.currentSLPage + 1);
    }
}

function previousSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.currentSLPage - 1);
    }
}

function goToSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderSLPage(pageNumber);
    }
}

function goToLastSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.slTotalPages);
    }
}

// Non-SL Gallery pagination
function nextNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.currentNonSLPage + 1);
    }
}

function previousNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.currentNonSLPage - 1);
    }
}

function goToNonSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderNonSLPage(pageNumber);
    }
}

function goToLastNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.nonSlTotalPages);
    }
}

// Testing SL Gallery pagination
function nextTestingSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingSLPage(galleryManager.currentTestingSLPage + 1);
    }
}

function previousTestingSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingSLPage(galleryManager.currentTestingSLPage - 1);
    }
}

function goToTestingSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderTestingSLPage(pageNumber);
    }
}

function goToLastTestingSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingSLPage(galleryManager.testingSlTotalPages);
    }
}

// Testing Non-SL Gallery pagination
function nextTestingNonSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingNonSLPage(galleryManager.currentTestingNonSLPage + 1);
    }
}

function previousTestingNonSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingNonSLPage(galleryManager.currentTestingNonSLPage - 1);
    }
}

function goToTestingNonSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderTestingNonSLPage(pageNumber);
    }
}

function goToLastTestingNonSLPage() {
    if (galleryManager) {
        galleryManager.renderTestingNonSLPage(galleryManager.testingNonSlTotalPages);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    galleryManager = new GalleryManager();
});
