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

                // Update active state
                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });

    // Mock API call to check model status (Will be replaced with Flask fetch in Phase 3)
    checkModelStatus();
});

function checkModelStatus() {
    // Phase 2: Hardcoded status reflecting your project brief
    const statusData = {
        'hog': { available: false, message: 'Artifacts Missing' },
        'cnn': { available: false, message: 'Pending Training' },
        'resnet': { available: false, message: 'Pending Training' }
    };

    updateStatusUI('status-hog', statusData.hog);
    updateStatusUI('status-cnn', statusData.cnn);
    updateStatusUI('status-resnet', statusData.resnet);
}

function updateStatusUI(elementId, status) {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.className = 'status-pill ' + (status.available ? 'active' : 'unavailable');
    el.querySelector('.status-text').textContent = status.message;
}