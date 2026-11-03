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

    // Handle File Selection
    fileInput.addEventListener('change', function (e) {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    // Handle Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation(); // Prevent triggering file input click
        resetUpload();
    });

    // Form Submission
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!currentFile) return;
        runMockPredictions();
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

        // Reset result cards
        document.querySelectorAll('.result-class').forEach(el => el.textContent = '—');
        document.querySelectorAll('.progress-bar-fill').forEach(el => el.style.width = '0%');
        document.querySelectorAll('.confidence-val').forEach(el => el.textContent = '—');
    }

    // Phase 2: Simulate Backend Call
    function runMockPredictions() {
        const originalText = predictBtn.querySelector('.btn-text').textContent;
        predictBtn.querySelector('.btn-text').textContent = 'Processing...';
        predictBtn.disabled = true;

        // Simulate network delay
        setTimeout(() => {
            // Mock output update
            updatePredictionCard('card-hog', 'No Model', 0);
            updatePredictionCard('card-cnn', 'Pending', 0);
            updatePredictionCard('card-resnet', 'Pending', 0);

            predictBtn.querySelector('.btn-text').textContent = originalText;
            predictBtn.disabled = false;

            document.getElementById('recommendation-container').innerHTML = `
                <p><strong>System Note:</strong> Inference simulated. Waiting for Flask backend (Phase 3) and actual model artifacts to process the image.</p>
            `;
        }, 1500);
    }

    function updatePredictionCard(cardId, className, confidence) {
        const card = document.getElementById(cardId);
        if (!card) return;
        card.querySelector('.result-class').textContent = className;
        card.querySelector('.confidence-val').textContent = confidence > 0 ? (confidence * 100).toFixed(1) + '%' : 'N/A';
        card.querySelector('.progress-bar-fill').style.width = (confidence * 100) + '%';
    }
});