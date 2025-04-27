const ctx = document.getElementById("driftChart").getContext('2d');

const driftChart = new Chart(ctx, {  // <--- Correct here
    type: "line",
    data: {
        labels: [], // timestamps
        datasets: [{
            label: "Drift Score Normalized",
            data: [],
            borderColor: "blue",
            borderWidth: 2,
            fill: false,
            tension: 0.3 // curve smoothness
        }]
    },
    options: {
        responsive: true,
        animation: false,
        scales: {
            x: {
                title: {
                    display: true,
                    text: "Timestep"
                }
            },
            y: {
                title: {
                    display: true,
                    text: "Normalized Drift Score"
                }
            }
        },
        plugins: {
            tooltip: {
                mode: "index",
                intersect: false,
            },
            legend: {
                display: true,
                position: "top",
            }
        },
    }
});

let currentIndex = 0;
let lastLength = null;
let sameLengthCounter = 0;
let monitoring = true;

const intervalId = setInterval(async () => {
    if (!monitoring) {
        console.log("Monitoring finished.");
        clearInterval(intervalId);
        return;
    }

    try {
        const res = await fetch(`/current_drift_score?i=${currentIndex}`);
        const json = await res.json();

        if (json.error) {
            console.error("Error from backend:", json.error);
            return;
        }

        const length = json.length;
        const body = json.body;

        if (body && body.timestamp !== undefined) {
            driftChart.data.labels.push(body.timestamp);
            driftChart.data.datasets[0].data.push(body.drift_score_normalized);
            driftChart.update();
            currentIndex += 1;
        }

        if (lastLength === length) {  // <--- Fix here
            sameLengthCounter += 1;
            console.log(`Same length ${length} detected ${sameLengthCounter} times`);
        } else {
            sameLengthCounter = 0;
        }

        lastLength = length;

        if (sameLengthCounter >= 3 && currentIndex >= lastLength) {
            monitoring = false;
        }

    } catch (err) {
        console.error("Error fetching drift score: ", err);
    }
}, 100);
