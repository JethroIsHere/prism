document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('accuracyChart').getContext('2d');

    // 1. Initialize Chart with empty baseline configuration
    const chartConfig = {
        type: 'bar',
        data: {
            labels: ['HOG + SVM', 'Custom CNN', 'ResNet-18'],
            datasets: [{
                label: 'Top-1 Accuracy (%)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(255, 255, 255, 0.25)', // Steel White for HOG
                    'rgba(255, 255, 255, 0.65)', // Bright White for Custom CNN
                    'rgba(233, 4, 19, 0.9)'      // PRISM Crimson for ResNet-18
                ],
                borderColor: [
                    '#ffffff',
                    '#ffffff',
                    '#E90413'
                ],
                borderWidth: 1,
                borderRadius: 0
            }]
        },
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

    window.benchmarkChart = new Chart(ctx, chartConfig);

    // 2. Fetch real data from Flask backend
    fetchMetricsAndUpdateUI();
});

async function fetchMetricsAndUpdateUI() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Failed to fetch metrics');

        const data = await response.json();

        // 3. Update Chart.js Data Arrays (using top1_accuracy_pct directly)
        const chartData = [
            data.hog ? data.hog.top1_accuracy_pct : 0,
            data.cnn ? data.cnn.top1_accuracy_pct : 0,
            data.resnet ? data.resnet.top1_accuracy_pct : 0
        ];

        window.benchmarkChart.data.datasets[0].data = chartData;
        window.benchmarkChart.update();

        // 4. Update HTML Table Cells
        updateTableRow('hog', data.hog);
        updateTableRow('cnn', data.cnn);
        updateTableRow('resnet', data.resnet);

        // 5. Update System Recommendation
        updateSystemRecommendation(data);

    } catch (error) {
        console.error('Error fetching metrics from backend:', error);
    }
}

function updateTableRow(modelPrefix, metrics) {
    if (!metrics) return;

    const safeSetText = (idSuffix, value) => {
        const el = document.getElementById(`${modelPrefix}-${idSuffix}`);
        if (el) el.textContent = value;
    };

    // Map exact JSON keys to the table
    safeSetText('acc', metrics.top1_accuracy_pct.toFixed(2) + '%');
    safeSetText('f1', metrics.macro_f1.toFixed(4));

    // Handle Ron's '_cpu' suffix vs Duke/Christine's standard runtime key
    const runtime = metrics.runtime_ms_per_image || metrics.runtime_ms_per_image_cpu;
    safeSetText('time', runtime + ' ms');

    safeSetText('params', metrics.parameter_count_millions + ' M');
    safeSetText('train', metrics.training_time_minutes + ' min');
}

function updateSystemRecommendation(data) {
    const container = document.getElementById('recommendation-container');

    if (!data.hog || !data.cnn || !data.resnet) {
        return;
    }

    const models = [
        { name: 'HOG + Linear SVM', stats: data.hog, type: 'classical' },
        { name: 'Custom CNN', stats: data.cnn, type: 'nn' },
        { name: 'ResNet-18', stats: data.resnet, type: 'pretrained' }
    ];

    // Crown the winner based on top1_accuracy_pct
    const bestModel = models.reduce((prev, current) => {
        return (prev.stats.top1_accuracy_pct > current.stats.top1_accuracy_pct) ? prev : current;
    });

    const bestRuntime = bestModel.stats.runtime_ms_per_image || bestModel.stats.runtime_ms_per_image_cpu;
    const hogRuntime = data.hog.runtime_ms_per_image_cpu;

    container.innerHTML = `
        <div class="recommendation-result">
            <h4 style="color: #10b981; margin-bottom: 0.5rem;">🏆 Recommended Architecture: ${bestModel.name}</h4>
            <p style="margin-bottom: 0.5rem; color: #e2e8f0;">
                <strong>Why?</strong> With a Top-1 Accuracy of <strong>${bestModel.stats.top1_accuracy_pct.toFixed(2)}%</strong> 
                and a Macro F1 score of <strong>${bestModel.stats.macro_f1.toFixed(4)}</strong>, this model significantly outperformed 
                the other approaches on the CIFAR-10 test set.
            </p>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                <em>Trade-off Note:</em> While it takes ${bestRuntime} ms per image to run inference 
                (compared to HOG's ${hogRuntime} ms), the massive gain in spatial feature recognition makes 
                the computational overhead entirely justifiable for this pipeline.
            </p>
        </div>
    `;
}