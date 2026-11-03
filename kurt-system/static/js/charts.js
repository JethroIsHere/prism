document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('accuracyChart').getContext('2d');

    // Phase 2: Mock Chart Data (Using Ron's actual HOG accuracy)
    const accuracyData = {
        labels: ['HOG + SVM', 'Custom CNN', 'ResNet-18'],
        datasets: [{
            label: 'Top-1 Accuracy (%)',
            data: [52.06, 0, 0], // CNN and ResNet are pending
            backgroundColor: [
                'rgba(236, 72, 153, 0.8)', // pink for HOG
                'rgba(59, 130, 246, 0.2)', // dimmed blue for pending CNN
                'rgba(16, 185, 129, 0.2)'  // dimmed green for pending ResNet
            ],
            borderColor: [
                'rgba(236, 72, 153, 1)',
                'rgba(59, 130, 246, 0.5)',
                'rgba(16, 185, 129, 0.5)'
            ],
            borderWidth: 1,
            borderRadius: 4
        }]
    };

    const chartConfig = {
        type: 'bar',
        data: accuracyData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            if (context.raw === 0) return 'Pending Training';
                            return context.raw + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    };

    // Store chart instance globally so it can be updated in Phase 5
    window.benchmarkChart = new Chart(ctx, chartConfig);
});