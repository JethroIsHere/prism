document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('image-input');
    const dropzone = document.getElementById('dropzone');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-image-btn');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const predictSpinner = document.getElementById('predict-spinner');
    const errorBanner = document.getElementById('upload-error');
    const resultsAnchor = document.getElementById('inference-results') || document.querySelector('.results-header');

    let currentFile = null;
    let isExecutionComplete = false;

    const LABELS = {
        IDLE: 'Run Multi-Model Prediction',
        EXECUTING: 'Executing Inference...',
        COMPLETE: 'Execution Complete — Add Another Image'
    };

    function showError(msg) {
        if (!errorBanner) return alert(msg);
        errorBanner.textContent = `[ERROR] ${msg}`;
        errorBanner.classList.remove('hidden');
    }

    function clearError() {
        if (errorBanner) errorBanner.classList.add('hidden');
    }

    // Ensure clean initial button state on load
    btnText.textContent = LABELS.IDLE;
    predictBtn.disabled = true;

    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) handleFile(this.files[0]);
    });

    // Drag-and-Drop Support
    ['dragenter', 'dragover'].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            handleFile(dt.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    // Handle clicking the button when in "Execution Complete" mode
    predictBtn.addEventListener('click', (e) => {
        if (isExecutionComplete) {
            e.preventDefault();
            e.stopPropagation();
            resetUpload();
        }
    });

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (isExecutionComplete) {
            resetUpload();
            return;
        }
        if (!currentFile) return;
        runPredictions(currentFile);
    });

    function handleFile(file) {
        clearError();
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            showError('Invalid file format. Upload a JPG or PNG image.');
            return;
        }

        // Reset completion state if user drops a new image directly
        isExecutionComplete = false;
        predictBtn.classList.remove('btn-complete');
        btnText.textContent = LABELS.IDLE;

        currentFile = file;
        const reader = new FileReader();

        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
        };

        reader.readAsDataURL(file);
    }

    function resetUpload() {
        clearError();
        currentFile = null;
        isExecutionComplete = false;
        fileInput.value = '';
        imagePreview.src = '';

        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');

        predictBtn.classList.remove('btn-complete');
        if (predictSpinner) predictSpinner.classList.add('hidden');
        btnText.textContent = LABELS.IDLE;
        predictBtn.disabled = true;

        document.querySelectorAll('.result-class').forEach(el => el.textContent = '—');
        document.querySelectorAll('.progress-bar-fill').forEach(el => el.style.width = '0%');
        document.querySelectorAll('.confidence-val').forEach(el => el.textContent = '—');
    }

    async function runPredictions(file) {
        clearError();
        isExecutionComplete = false;
        predictBtn.classList.remove('btn-complete');
        btnText.textContent = LABELS.EXECUTING;
        if (predictSpinner) predictSpinner.classList.remove('hidden');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const results = await response.json();
            if (!response.ok) throw new Error(results.error || 'Prediction request failed');

            updatePredictionCard('card-hog', results.hog);
            updatePredictionCard('card-cnn', results.cnn);
            updatePredictionCard('card-resnet', results.resnet);
            updatePredictionCard('card-knn', results.knn);
            updatePredictionCard('card-efficientnet', results.efficientnet);

            // Transition button to Completion / Reset state
            isExecutionComplete = true;
            btnText.textContent = LABELS.COMPLETE;
            predictBtn.classList.add('btn-complete');
            predictBtn.disabled = false;

            // Smoothly slide down to Comparative Inference Output
            if (resultsAnchor) {
                resultsAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

        } catch (error) {
            console.error('Inference Error:', error);
            showError(error.message || 'Server connection interrupted.');
            btnText.textContent = LABELS.IDLE;
            predictBtn.disabled = false;
        } finally {
            if (predictSpinner) predictSpinner.classList.add('hidden');
        }
    }

    function updatePredictionCard(cardId, modelData) {
        const card = document.getElementById(cardId);
        if (!card || !modelData) return;

        const badge = card.querySelector('.status-badge');
        const resultClass = card.querySelector('.result-class');
        const confVal = card.querySelector('.confidence-val');
        const progressBar = card.querySelector('.progress-bar-fill');

        if (badge && modelData.status) {
            badge.textContent = modelData.status.message;
            badge.className = 'status-badge ' + (modelData.status.available ? 'active' : 'unavailable');
        }

        if (modelData.prediction) {
            resultClass.textContent = modelData.prediction;
            const pct = Math.min(100, Math.max(0, modelData.confidence * 100)).toFixed(1);
            confVal.textContent = `${pct}%`;
            progressBar.style.width = `${pct}%`;
        } else {
            resultClass.textContent = 'N/A';
            confVal.textContent = '—';
            progressBar.style.width = '0%';
        }
    }
});