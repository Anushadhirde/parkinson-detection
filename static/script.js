document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const errorMsg = document.getElementById('errorMsg');
    
    const emptyState = document.getElementById('emptyState');
    const previewState = document.getElementById('previewState');
    const audioPlayer = document.getElementById('audioPlayer');
    const loadingState = document.getElementById('loadingState');
    const resultState = document.getElementById('resultState');

    let currentFile = null;

    // Trigger file input dialog on dropzone click
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and Drop functionality
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary)';
        dropzone.style.backgroundColor = 'rgba(0, 128, 128, 0.08)';
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#b2d8d8';
        dropzone.style.backgroundColor = 'rgba(0, 128, 128, 0.03)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#b2d8d8';
        dropzone.style.backgroundColor = 'rgba(0, 128, 128, 0.03)';
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Clear file
    clearBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        
        // UI Reset
        dropzone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        errorMsg.classList.add('hidden');
        
        emptyState.classList.remove('hidden');
        previewState.classList.add('hidden');
        resultState.classList.add('hidden');
        loadingState.classList.add('hidden');
        
        audioPlayer.pause();
        audioPlayer.src = '';
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    function handleFile(file) {
        errorMsg.classList.add('hidden');
        
        // Validate
        if (!file.name.toLowerCase().endsWith('.wav')) {
            showError("Invalid file format. Please upload a .WAV file.");
            return;
        }

        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 200) {
            showError("File is too large. Maximum size is 200MB.");
            return;
        }

        currentFile = file;

        // UI Updates
        fileName.textContent = file.name;
        fileSize.textContent = sizeMB.toFixed(2) + ' MB';
        
        dropzone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');

        // Set Audio Preview
        const objectUrl = URL.createObjectURL(file);
        audioPlayer.src = objectUrl;
        
        emptyState.classList.add('hidden');
        previewState.classList.remove('hidden');
        resultState.classList.add('hidden');
    }

    // Analyze Button Logic
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show Loading State
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "<i class='bx bx-loader-alt bx-spin'></i> Analyzing...";
        previewState.classList.add('hidden');
        resultState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        errorMsg.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to process audio');
            }

            renderResults(data);

        } catch (error) {
            showError('An error occurred during prediction: ' + error.message);
            loadingState.classList.add('hidden');
            previewState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = "<i class='bx bx-search-alt-2'></i> Analyze Acoustic Signatures";
        }
    });

    function renderResults(data) {
        const result = data.result; // "healthy" or "parkinson"
        const confidence = data.confidence;
        
        const isHealthy = result === "healthy";
        const pulseClass = isHealthy ? "result-healthy" : "result-parkinson";
        const resultText = isHealthy ? "NEGATIVE FOR PARKINSON'S" : "POTENTIAL INDICATIONS DETECTED";
        const resultIcon = isHealthy ? "✓" : "⚠️";
        const percentage = Math.round(confidence * 100);
        
        // Calculate stroke Dasharray for SVG Circle
        const strokeArray = `${percentage}, 100`;

        const html = `
            <div class="result-pulse ${pulseClass}">
                <div class="result-icon">${resultIcon}</div>
                <div style="font-size: 1.4rem;">${resultText}</div>
                
                <svg viewBox="0 0 36 36" class="circular-chart">
                    <path class="circle-bg"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path class="circle"
                        stroke-dasharray="${strokeArray}"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <text x="18" y="20.35" class="percentage">${percentage}%</text>
                    <text x="18" y="24.5" class="percentage-label">Confidence</text>
                </svg>
            </div>
        `;

        resultState.innerHTML = html;
        loadingState.classList.add('hidden');
        resultState.classList.remove('hidden');
        previewState.classList.remove('hidden');
    }
});
