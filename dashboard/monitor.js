const ctx = document.getElementById("driftChart").getContext('2d');

const driftChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Drift Score Normalized",
            data: [],
            borderColor: "blue",
            borderWidth: 2,
            fill: false,
            tension: 0.3
        }]
    },
    options: {
        responsive: true,
        animation: false,
        scales: {
            x: { title: { display: true, text: "Timestep" } },
            y: { title: { display: true, text: "Normalized Drift Score" } }
        },
        plugins: {
            tooltip: { mode: "index", intersect: false },
            legend: { display: true, position: "top" }
        }
    }
});

// Connect to WebSocket server
const socket = new WebSocket(`ws://${window.location.host}/ws/response`);

socket.addEventListener('message', function(event) {
    const data = JSON.parse(event.data);
    if (data && data.timestamp !== undefined) {
        driftChart.data.labels.push(data.timestamp);
        driftChart.data.datasets[0].data.push(data.drift_score_normalized);
        driftChart.update();
    }
});
