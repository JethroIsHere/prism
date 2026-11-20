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
        updateStatusUI('status-knn', statusData.knn);
        updateStatusUI('status-efficientnet', statusData.efficientnet);
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

// Scroll-Spy for Navbar Active State & Thumb-Tracking Scroll Percentage Marker
document.addEventListener('DOMContentLoaded', () => {
    const sections = ['dashboard', 'inference', 'benchmark', 'about']
        .map(id => document.getElementById(id))
        .filter(Boolean);
    const navLinks = document.querySelectorAll('.nav-links a');
    const scrollTelemetryEl = document.getElementById('scroll-telemetry');
    const scrollPctEl = document.getElementById('scroll-pct');
    let scrollHideTimer = null;

    function updateScrollTelemetry(isUserScrolling = false) {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight;
        const winHeight = window.innerHeight;
        const scrollableDistance = docHeight - winHeight;

        const progress = scrollableDistance > 0
            ? Math.min(1, Math.max(0, scrollTop / scrollableDistance))
            : 0;
        const pct = Math.round(progress * 100);

        if (scrollPctEl) {
            scrollPctEl.textContent = `${String(pct).padStart(2, '0')}%`;
        }

        // Align pill vertically with the center of the red scrollbar thumb
        if (scrollTelemetryEl && scrollableDistance > 0) {
            const arrowOffset = 16; // Accounts for browser top/bottom scrollbar arrow buttons
            const trackHeight = winHeight - arrowOffset * 2;
            const thumbHeight = Math.max(36, (winHeight / docHeight) * trackHeight);
            const maxThumbTravel = Math.max(0, trackHeight - thumbHeight);
            const thumbCenterY = arrowOffset + (progress * maxThumbTravel) + (thumbHeight / 2);

            scrollTelemetryEl.style.top = `${thumbCenterY}px`;

            if (isUserScrolling) {
                scrollTelemetryEl.classList.add('is-scrolling');
                clearTimeout(scrollHideTimer);
                scrollHideTimer = setTimeout(() => {
                    scrollTelemetryEl.classList.remove('is-scrolling');
                }, 850);
            }
        }

        // Highlight the navbar tab currently visible in viewport
        const triggerLine = scrollTop + winHeight * 0.35;
        let currentId = sections[0]?.id;

        for (const section of sections) {
            if (triggerLine >= section.offsetTop) {
                currentId = section.id;
            }
        }

        if (currentId) {
            navLinks.forEach(link => {
                const isMatch = link.getAttribute('href') === `#${currentId}`;
                link.classList.toggle('active', isMatch);
            });
        }
    }

    window.addEventListener('scroll', () => updateScrollTelemetry(true), { passive: true });
    window.addEventListener('resize', () => updateScrollTelemetry(false), { passive: true });
    updateScrollTelemetry(false);
});