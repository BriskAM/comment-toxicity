document.addEventListener("DOMContentLoaded", () => {
  Chart.defaults.color = "#9caa99";
  Chart.defaults.borderColor = "rgba(200, 220, 200, 0.12)";
  Chart.defaults.font.family = "Inter, ui-sans-serif, system-ui, sans-serif";

  const green = "#4ade80";
  const amber = "#fb923c";
  const red = "#ef4444";
  const purple = "#a855f7";
  const blue = "#60a5fa";
  const bg = "#111c13";

  new Chart(document.getElementById("classDistributionChart"), {
    type: "bar",
    data: {
      labels: ["0 (Non-toxic)", "1 (Toxic)", "2 (Very toxic)", "3 (Extremely)"],
      datasets: [
        {
          label: "Count",
          data: [114173, 15918, 62440, 5469],
          backgroundColor: [green, amber, red, purple],
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Label Distribution", font: { size: 15, weight: "bold" }, padding: 20 },
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw.toLocaleString()} samples (${(ctx.raw / 198000 * 100).toFixed(1)}%)` } },
      },
      scales: {
        y: { title: { display: true, text: "Sample Count", font: { size: 12 } }, grid: { color: "rgba(200,220,200,0.06)" } },
        x: { ticks: { maxRotation: 25 } },
      },
    },
  });

  new Chart(document.getElementById("demographicChart"), {
    type: "bar",
    data: {
      labels: ["Non-toxic (0)", "Toxic (1)", "Very toxic (2)", "Extremely (3)"],
      datasets: [
        {
          label: "Avg demographic fields filled",
          data: [0.62, 2.49, 0.70, 0.61],
          backgroundColor: [green, amber, red, purple],
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Avg Demographic Fields Filled by Toxicity Label", font: { size: 13, weight: "bold" }, padding: 16 },
        legend: { display: false },
      },
      scales: {
        y: { title: { display: true, text: "Fields Filled (0-3)" }, grid: { color: "rgba(200,220,200,0.06)" } },
      },
    },
  });

  new Chart(document.getElementById("engagementChart"), {
    type: "bar",
    data: {
      labels: ["Non-toxic (0)", "Toxic (1)", "Very toxic (2)", "Extremely (3)"],
      datasets: [
        { label: "Avg Upvotes", data: [3.12, 5.47, 2.94, 2.18], backgroundColor: green, borderRadius: 4 },
        { label: "Avg If_1", data: [1.84, 8.92, 2.15, 1.67], backgroundColor: amber, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Engagement Metrics by Label", font: { size: 13, weight: "bold" }, padding: 16 },
        legend: { labels: { usePointStyle: true, pointStyleWidth: 8 } },
      },
      scales: {
        y: { grid: { color: "rgba(200,220,200,0.06)" } },
      },
    },
  });

  new Chart(document.getElementById("modelComparisonChart"), {
    type: "bar",
    data: {
      labels: ["LightGBM\n(Tuned)", "LightGBM", "Random Forest\n(Subsampled)", "Logistic\nRegression"],
      datasets: [
        { label: "Accuracy", data: [0.912, 0.913, 0.825, 0.725], backgroundColor: green, borderRadius: 4 },
        { label: "Macro F1", data: [0.817, 0.813, 0.587, 0.494], backgroundColor: amber, borderRadius: 4 },
        { label: "Weighted F1", data: [0.913, 0.913, 0.823, 0.698], backgroundColor: blue, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Model Comparison (OOF Predictions)", font: { size: 15, weight: "bold" }, padding: 20 },
        legend: { labels: { usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}` } },
      },
      scales: {
        y: { min: 0, max: 1, title: { display: true, text: "Score", font: { size: 12 } }, grid: { color: "rgba(200,220,200,0.06)" } },
      },
    },
  });

  new Chart(document.getElementById("perClassChart"), {
    type: "bar",
    data: {
      labels: ["Class 0\n(Non-toxic)", "Class 1\n(Toxic)", "Class 2\n(Very toxic)", "Class 3\n(Extremely)"],
      datasets: [
        { label: "Precision", data: [0.98, 0.76, 0.86, 0.66], backgroundColor: green, borderRadius: 4 },
        { label: "Recall", data: [0.95, 0.81, 0.90, 0.61], backgroundColor: amber, borderRadius: 4 },
        { label: "F1 Score", data: [0.96, 0.79, 0.88, 0.64], backgroundColor: blue, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Per-Class Performance (Threshold-Tuned LightGBM)", font: { size: 14, weight: "bold" }, padding: 18 },
        legend: { labels: { usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}` } },
      },
      scales: {
        y: { min: 0, max: 1.05, grid: { color: "rgba(200,220,200,0.06)" }, title: { display: true, text: "Score", font: { size: 12 } } },
      },
    },
  });

  function renderConfusionHeatmap() {
    const canvas = document.getElementById("confusionHeatmap");
    const ctx = canvas.getContext("2d");
    const cm = [
      [108464, 1520, 4075, 114],
      [1716, 12893, 1212, 97],
      [846, 2080, 56803, 2711],
      [2, 480, 1902, 3085],
    ];
    const labels = ["Pred 0", "Pred 1", "Pred 2", "Pred 3"];
    const trueLabels = ["True 0", "True 1", "True 2", "True 3"];
    const cellSize = 50;
    const pad = 50;
    const w = pad + 4 * cellSize + 80;
    const h = pad + 4 * cellSize + 40;
    canvas.width = w * 2;
    canvas.height = h * 2;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    ctx.scale(2, 2);

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const maxVal = Math.max(...cm.flat());
    const getColor = (v) => {
      const t = v / maxVal;
      const r = Math.round(10 + (245 * t));
      const g = Math.round(15 + (150 * (1 - t)));
      const b = Math.round(11 + (40 * (1 - t)));
      return `rgb(${r},${g},${b})`;
    };

    cm.forEach((row, i) => {
      row.forEach((val, j) => {
        ctx.fillStyle = getColor(val);
        ctx.fillRect(pad + j * cellSize, pad + i * cellSize, cellSize, cellSize);
        ctx.fillStyle = val / maxVal > 0.5 ? "#0a0f0b" : "#edf6e9";
        ctx.font = "bold 12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(val.toLocaleString(), pad + j * cellSize + cellSize / 2, pad + i * cellSize + cellSize / 2);
      });
    });

    ctx.fillStyle = "#9caa99";
    ctx.font = "11px Inter, sans-serif";
    labels.forEach((l, j) => {
      ctx.fillText(l, pad + j * cellSize + cellSize / 2, pad - 10);
    });
    trueLabels.forEach((l, i) => {
      ctx.textAlign = "right";
      ctx.fillText(l, pad - 8, pad + i * cellSize + cellSize / 2);
    });

    ctx.fillStyle = "#9caa99";
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Predicted Label", pad + 2 * cellSize, pad - 30);
    ctx.save();
    ctx.translate(pad - 34, pad + 2 * cellSize + 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Actual Label", 0, 0);
    ctx.restore();
  }

  renderConfusionHeatmap();

  new Chart(document.getElementById("featureImportanceChart"), {
    type: "bar",
    data: {
      labels: [
        "post_label_3_rate", "post_label_1_rate", "demo_count", "post_label_2_rate",
        "race_p1", "gender_p1", "religion_p1", "if_1", "upvote",
        "word_count", "char_count", "caps_ratio", "post_comment_position",
        "exclamation_count", "has_url",
      ],
      datasets: [
        {
          label: "Gain",
          data: [2850, 2410, 1980, 1750, 1520, 1380, 1240, 1100, 950, 820, 730, 640, 580, 510, 450],
          backgroundColor: [
            red, red, amber, amber, amber, amber, amber,
            green, green, green, green, green, green, green, green,
          ],
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        title: { display: true, text: "Top 15 Features — LightGBM (Gain)", font: { size: 14, weight: "bold" }, padding: 18 },
        legend: { display: false },
      },
      scales: {
        x: { title: { display: true, text: "Feature Importance (Gain)", font: { size: 12 } }, grid: { color: "rgba(200,220,200,0.06)" } },
        y: { ticks: { font: { size: 10 } } },
      },
    },
  });
});
