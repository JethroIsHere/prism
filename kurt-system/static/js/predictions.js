document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('image-input');
    const dropzone = document.getElementById('dropzone');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-image-btn');
    const predictBtn = document.getElementById('predict-btn');
    const uploadForm = document.getElementById('upload-form');

    let currentFile = null;

    fileInput.addEventListener('change', function (e) {
        if (this.files && this.files[0]) handleFile(this.files[0]);
    });

    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!currentFile) return;
        runPredictions(currentFile);
    });

    function handleFile(file) {
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Please upload a valid JPG or PNG image.');
            return;
        }

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
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        predictBtn.disabled = true;

        document.querySelectorAll('.result-class').forEach(el => el.textContent = '—');
        document.querySelectorAll('.progress-bar-fill').forEach(el => el.style.width = '0%');
        document.querySelectorAll('.confidence-val').forEach(el => el.textContent = '—');
        document.getElementById('recommendation-container').innerHTML = `
            <div class="empty-state">
                <p>Upload an image and run inference or check the metrics dashboard to see the system recommendation based on actual evaluation metrics.</p>
            </div>`;
    }

    async function runPredictions(file) {
        const originalText = predictBtn.querySelector('.btn-text').textContent;
        predictBtn.querySelector('.btn-text').textContent = 'Processing Image...';
        predictBtn.disabled = true;

        // Prepare file for HTTP POST
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Prediction request failed');

            const results = await response.json();

            // Update UI with backend results
            updatePredictionCard('card-hog', results.hog);
            updatePredictionCard('card-cnn', results.cnn);
            updatePredictionCard('card-resnet', results.resnet);

            document.getElementById('recommendation-container').innerHTML = `
                <p><strong>System Note:</strong> Phase 3 Backend connected successfully. Upload routed to prediction API.</p>
            `;

        } catch (error) {
            console.error('Inference Error:', error);
            alert('An error occurred while communicating with the server.');
        } finally {
            predictBtn.querySelector('.btn-text').textContent = originalText;
            predictBtn.disabled = false;
        }
    }

    function updatePredictionCard(cardId, modelData) {
        const card = document.getElementById(cardId);
        if (!card) return;

        const statusBadge = card.querySelector('.status-badge');
        statusBadge.textContent = modelData.status.message;

        // Update styling based on availability
        if (modelData.status.available) {
            statusBadge.className = 'status-badge active';
        } else if (modelData.status.message === 'Pending Training') {
            statusBadge.className = 'status-badge pending';
        } else {
            statusBadge.className = 'status-badge unavailable';
        }

        const className = modelData.prediction ? modelData.prediction : '—';
        card.querySelector('.result-class').textContent = className;

        const confidenceStr = modelData.confidence > 0 ? (modelData.confidence * 100).toFixed(1) + '%' : 'N/A';
        card.querySelector('.confidence-val').textContent = confidenceStr;
        card.querySelector('.progress-bar-fill').style.width = (modelData.confidence * 100) + '%';
    }
});