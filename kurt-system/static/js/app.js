document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);

            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 80,
                    behavior: 'smooth'
                });

                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });

    // Fetch real model status from Flask Backend
    fetchModelStatus();
});

async function fetchModelStatus() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error('Failed to fetch model status');

        const statusData = await response.json();

        updateStatusUI('status-hog', statusData.hog);
        updateStatusUI('status-cnn', statusData.cnn);
        updateStatusUI('status-resnet', statusData.resnet);
    } catch (error) {
        console.error('Error fetching model statuses:', error);
        // Fallback UI if backend is unreachable
        updateStatusUI('status-hog', { available: false, message: 'Backend Disconnected' });
        updateStatusUI('status-cnn', { available: false, message: 'Backend Disconnected' });
        updateStatusUI('status-resnet', { available: false, message: 'Backend Disconnected' });
    }
}

function updateStatusUI(elementId, status) {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.className = 'status-pill ' + (status.available ? 'active' : 'unavailable');
    if (!status.available && status.message === 'Pending Training') {
        el.className = 'status-pill pending';
    }
    el.querySelector('.status-text').textContent = status.message;
}