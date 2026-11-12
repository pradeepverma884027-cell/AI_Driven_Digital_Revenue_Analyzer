const API_BASE = "";

let scenarioData = [];
let currentScenario = "Baseline";

document.addEventListener("DOMContentLoaded", () => {
    initializeNavigation();
    loadForecastSummary();
    loadForecast();
    loadScenarios();
    loadDrivers();
    loadInsights();
});

function formatCurrency(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "$0";
    }

    return "$" + number.toLocaleString("en-US", {
        maximumFractionDigits: 0
    });
}

function formatROAS(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "0.00x";
    }

    return number.toFixed(2) + "x";
}

function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}

function initializeNavigation() {
    const navItems = document.querySelectorAll(".nav-item");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            navItems.forEach(nav => {
                nav.classList.remove("active");
            });

            item.classList.add("active");
        });
    });
}

async function loadForecastSummary() {
    try {
        const response = await fetch(`${API_BASE}/forecast/summary`);

        if (!response.ok) {
            throw new Error("Forecast summary request failed");
        }

        const data = await response.json();

        if (data.forecast && Array.isArray(data.forecast)) {
            data.forecast.forEach(item => {
                const days = Number(item.horizon_days);

                if (days === 30) {
                    setText(
                        "revenue30",
                        formatCurrency(item.p50_revenue)
                    );
                }

                if (days === 60) {
                    setText(
                        "revenue60",
                        formatCurrency(item.p50_revenue)
                    );
                }

                if (days === 90) {
                    setText(
                        "revenue90",
                        formatCurrency(item.p50_revenue)
                    );
                }
            });
        }

        loadROAS();

    } catch (error) {
        console.error(error);

        setText("revenue30", "Error");
        setText("revenue60", "Error");
        setText("revenue90", "Error");
    }
}

async function loadROAS() {
    try {
        const response = await fetch(`${API_BASE}/roas`);

        if (!response.ok) {
            throw new Error("ROAS request failed");
        }

        const data = await response.json();

        if (!Array.isArray(data) || data.length === 0) {
            setText("roas90", "--");
            return;
        }

        const rows = data.slice(0, 90);

        let revenue = 0;
        let spend = 0;

        rows.forEach(row => {
            const revenueValue = Number(
                row.P50_Revenue ??
                row.P50 ??
                row.Forecast_Revenue ??
                row.revenue ??
                0
            );

            const spendValue = Number(
                row.Spend ??
                row.Forecast_Spend ??
                row.spend ??
                0
            );

            if (Number.isFinite(revenueValue)) {
                revenue += revenueValue;
            }

            if (Number.isFinite(spendValue)) {
                spend += spendValue;
            }
        });

        if (spend > 0) {
            setText(
                "roas90",
                formatROAS(revenue / spend)
            );
        } else {
            setText("roas90", "--");
        }

    } catch (error) {
        console.error(error);
        setText("roas90", "--");
    }
}

async function loadForecast() {
    try {
        const response = await fetch(`${API_BASE}/forecast`);

        if (!response.ok) {
            throw new Error("Forecast request failed");
        }

        const data = await response.json();

        renderForecastChart(data);

    } catch (error) {
        console.error(error);

        const chart = document.getElementById("forecastChart");

        if (chart) {
            chart.textContent = "Unable to load forecast.";
        }
    }
}

function renderForecastChart(data) {
    const container = document.getElementById("forecastChart");

    if (!container) {
        return;
    }

    if (!Array.isArray(data) || data.length === 0) {
        container.textContent = "No forecast data available.";
        return;
    }

    const rows = data.slice(0, 90);

    const width = 1000;
    const height = 310;
    const padding = 40;

    const values = [];

    rows.forEach(row => {
        values.push(Number(row.P10) || 0);
        values.push(Number(row.P50) || 0);
        values.push(Number(row.P90) || 0);
    });

    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    const range = maxValue - minValue || 1;

    function x(index) {
        return padding +
            (index / Math.max(rows.length - 1, 1)) *
            (width - padding * 2);
    }

    function y(value) {
        return height -
            padding -
            ((value - minValue) / range) *
            (height - padding * 2);
    }

    function createPath(column) {
        return rows.map((row, index) => {
            const px = x(index);
            const py = y(Number(row[column]) || 0);

            return index === 0
                ? `M ${px} ${py}`
                : `L ${px} ${py}`;
        }).join(" ");
    }

    const p10Path = createPath("P10");
    const p50Path = createPath("P50");
    const p90Path = createPath("P90");

    container.innerHTML = `
        <svg
            class="forecast-svg"
            viewBox="0 0 ${width} ${height}"
            preserveAspectRatio="none"
        >
            <line
                x1="${padding}"
                y1="${height - padding}"
                x2="${width - padding}"
                y2="${height - padding}"
                stroke="#e5e7eb"
            />

            <line
                x1="${padding}"
                y1="${padding}"
                x2="${padding}"
                y2="${height - padding}"
                stroke="#e5e7eb"
            />

            <path
                d="${p90Path}"
                fill="none"
                stroke="#94a3b8"
                stroke-width="2"
                stroke-dasharray="5 5"
            />

            <path
                d="${p10Path}"
                fill="none"
                stroke="#cbd5e1"
                stroke-width="2"
                stroke-dasharray="5 5"
            />

            <path
                d="${p50Path}"
                fill="none"
                stroke="#111827"
                stroke-width="3"
            />

            <text
                x="${padding}"
                y="20"
                font-size="11"
                fill="#9ca3af"
            >
                Revenue Forecast
            </text>

            <text
                x="${width - padding}"
                y="${height - 10}"
                text-anchor="end"
                font-size="10"
                fill="#9ca3af"
            >
                90 Days
            </text>
        </svg>
    `;
}

async function loadScenarios() {
    const status = document.getElementById("scenarioStatus");
    const selector = document.getElementById("scenarioSelect");

    try {
        if (status) {
            status.textContent = "Loading scenarios...";
        }

        const response = await fetch(`${API_BASE}/scenarios`);

        if (!response.ok) {
            throw new Error(
                `Scenario API returned ${response.status}`
            );
        }

        const data = await response.json();

        console.log("Scenario data:", data);

        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("No scenario data returned");
        }

        scenarioData = data;

        const scenarios = [
            ...new Set(
                data
                    .map(row => String(row.Scenario).trim())
                    .filter(Boolean)
            )
        ];

        if (!selector) {
            throw new Error("Scenario selector not found");
        }

        selector.innerHTML = "";

        scenarios.forEach(scenario => {
            const option = document.createElement("option");

            option.value = scenario;
            option.textContent = scenario;

            if (
                scenario.toLowerCase() ===
                currentScenario.toLowerCase()
            ) {
                option.selected = true;
            }

            selector.appendChild(option);
        });

        const matchingScenario = scenarios.find(
            scenario =>
                scenario.toLowerCase() ===
                currentScenario.toLowerCase()
        );

        if (matchingScenario) {
            currentScenario = matchingScenario;
        } else if (scenarios.length > 0) {
            currentScenario = scenarios[0];
            selector.value = currentScenario;
        }

        selector.removeEventListener(
            "change",
            handleScenarioChange
        );

        selector.addEventListener(
            "change",
            handleScenarioChange
        );

        updateScenario(currentScenario);

        if (status) {
            status.textContent =
                `${scenarioData.length} scenario forecasts loaded.`;
        }

    } catch (error) {
        console.error("Scenario error:", error);

        if (status) {
            status.textContent =
                "Unable to load scenario forecasts.";
        }

        if (selector) {
            selector.disabled = true;
        }
    }
}

function handleScenarioChange(event) {
    currentScenario = event.target.value;

    console.log(
        "Selected scenario:",
        currentScenario
    );

    updateScenario(currentScenario);
}

function updateScenario(scenarioName) {
    if (!Array.isArray(scenarioData)) {
        return;
    }

    const selectedRows = scenarioData.filter(row => {
        return (
            String(row.Scenario)
                .trim()
                .toLowerCase() ===
            String(scenarioName)
                .trim()
                .toLowerCase()
        );
    });

    console.log(
        "Rows for",
        scenarioName,
        selectedRows
    );

    if (selectedRows.length === 0) {
        console.error(
            "No scenario rows found:",
            scenarioName
        );
        return;
    }

    const row30 = selectedRows.find(
        row => Number(row.Horizon_Days) === 30
    );

    const row60 = selectedRows.find(
        row => Number(row.Horizon_Days) === 60
    );

    const row90 = selectedRows.find(
        row => Number(row.Horizon_Days) === 90
    );

    updateScenarioRow(
        row30,
        "scenarioRevenue30",
        "scenarioSpend30",
        "scenarioRoas30"
    );

    updateScenarioRow(
        row60,
        "scenarioRevenue60",
        "scenarioSpend60",
        "scenarioRoas60"
    );

    updateScenarioRow(
        row90,
        "scenarioRevenue90",
        "scenarioSpend90",
        "scenarioRoas90"
    );

    const status =
        document.getElementById("scenarioStatus");

    if (status) {
        status.textContent =
            `Showing ${scenarioName} scenario forecast.`;
    }
}

function updateScenarioRow(
    row,
    revenueId,
    spendId,
    roasId
) {
    if (!row) {
        return;
    }

    setText(
        revenueId,
        formatCurrency(
            row.Forecast_Revenue
        )
    );

    setText(
        spendId,
        formatCurrency(
            row.Forecast_Spend
        )
    );

    setText(
        roasId,
        formatROAS(
            row.Forecast_ROAS
        )
    );
}

async function loadDrivers() {
    try {
        const response = await fetch(
            `${API_BASE}/drivers`
        );

        if (!response.ok) {
            throw new Error(
                "Drivers request failed"
            );
        }

        const data = await response.json();

        renderDrivers(data);

    } catch (error) {
        console.error(error);

        const container =
            document.getElementById(
                "driversList"
            );

        if (container) {
            container.textContent =
                "Unable to load drivers.";
        }
    }
}

function renderDrivers(drivers) {
    const container =
        document.getElementById(
            "driversList"
        );

    if (!container) {
        return;
    }

    if (
        !Array.isArray(drivers) ||
        drivers.length === 0
    ) {
        container.textContent =
            "No driver data available.";
        return;
    }

    const topDrivers =
        drivers.slice(0, 5);

    const maxImportance =
        Math.max(
            ...topDrivers.map(
                driver =>
                    Number(
                        driver.Importance
                    ) || 0
            )
        );

    container.innerHTML = "";

    topDrivers.forEach(driver => {
        const importance =
            Number(driver.Importance) || 0;

        const percentage =
            maxImportance > 0
                ? (importance / maxImportance) * 100
                : 0;

        const row =
            document.createElement("div");

        row.className =
            "driver-row";

        row.innerHTML = `
            <div class="driver-rank">
                ${driver.Rank ?? ""}
            </div>

            <div>
                <div class="driver-name">
                    ${driver.Feature ?? "Unknown"}
                </div>

                <div class="driver-bar">
                    <div
                        class="driver-bar-inner"
                        style="width:${percentage}%"
                    ></div>
                </div>
            </div>

            <div class="driver-value">
                ${importance.toFixed(4)}
            </div>
        `;

        container.appendChild(row);
    });
}

async function loadInsights() {
    try {
        const response = await fetch(
            `${API_BASE}/insights`
        );

        if (!response.ok) {
            throw new Error(
                "Insights request failed"
            );
        }

        const data = await response.json();

        renderInsights(data);

    } catch (error) {
        console.error(error);

        const container =
            document.getElementById(
                "insightsCard"
            );

        if (container) {
            container.textContent =
                "Unable to load insights.";
        }
    }
}

function renderInsights(data) {
    const container =
        document.getElementById(
            "insightsCard"
        );

    if (!container) {
        return;
    }

    const topDriver =
        data.top_drivers &&
        data.top_drivers.length > 0
            ? data.top_drivers[0].Feature
            : "revenue drivers";

    container.innerHTML = `
        <div class="insight-title">
            AI-GENERATED INSIGHT
        </div>

        <div class="insight-main">
            Revenue is primarily influenced by
            <strong>${topDriver}</strong>.
        </div>

        <div class="insight-action">
            <strong>Recommended action:</strong>
            Monitor the leading revenue drivers and
            adjust marketing investment based on the
            selected forecast scenario.
        </div>
    `;

    renderAssumptions(
        data.assumptions
    );
}

function renderAssumptions(assumptions) {
    const container =
        document.getElementById(
            "assumptions"
        );

    if (!container) {
        return;
    }

    if (
        !Array.isArray(assumptions) ||
        assumptions.length === 0
    ) {
        container.textContent =
            "No assumptions available.";
        return;
    }

    container.innerHTML = "";

    assumptions.forEach(assumption => {
        const item =
            document.createElement("div");

        item.className =
            "assumption-item";

        item.textContent =
            assumption;

        container.appendChild(item);
    });
}