const state = {
  summary: null,
  houses: [],
};

const tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);

const map = L.map('map', { preferCanvas: true }).setView([42.03, -93.63], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

//L.marker([42.03, -93.63]).addTo(map);

let layerGroup = L.featureGroup().addTo(map);

// blue marker icon
const blueIcon = L.icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",

  iconSize: [12, 20],     // smaller pin
  iconAnchor: [6, 20],    // tip of pin
  popupAnchor: [0, -18],  // popup position
  shadowSize: [20, 20]
});

function currency(v) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);
}

function buildQuery() {
  const params = new URLSearchParams();
  const pairs = {
    neighborhood: document.getElementById('neighborhood').value,
    metric: document.getElementById('metric').value,
    min_price: document.getElementById('minPrice').value,
    max_price: document.getElementById('maxPrice').value,
    min_year: document.getElementById('minYear').value,
    max_year: document.getElementById('maxYear').value,
    min_qual: document.getElementById('minQual').value,
  };
  Object.entries(pairs).forEach(([k, v]) => {
    if (v && v !== 'All') params.set(k, v);

  });
  return params.toString();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    console.error("API error:", url);
    return [];
  }
  return res.json();
}

function drawMetricCards() {

  console.log("drawMetricCards running");

  if (!state.summary) {
    console.log("summary not loaded yet");
    return;
  }
  const c = document.getElementById('metric-cards');
  const rf = state.summary.model_metrics.random_forest;
  c.innerHTML = `
    <div class="metric-card"><div class="label">Homes</div><div class="value">${state.summary.row_count}</div></div>
    <div class="metric-card"><div class="label">Mean Price</div><div class="value">${currency(state.summary.price_mean)}</div></div>
    <div class="metric-card"><div class="label">RF R²</div><div class="value">${rf.r2}</div></div>
    <div class="metric-card"><div class="label">RF RMSE</div><div class="value">${currency(rf.rmse)}</div></div>
  `;
}

function initControls(summary) {
  const neighborhood = document.getElementById('neighborhood');
  neighborhood.innerHTML = '<option>All</option>' + summary.neighborhoods.map(d => `<option value="${d}">${d}</option>`).join('');

  document.getElementById('minYear').value = summary.year_min;
  document.getElementById('maxYear').value = summary.year_max;
  document.getElementById('minQual').value = summary.quality_min;
}

function getColorScale(metric, values) {
  if (metric === 'Cluster') {
    return d3.scaleOrdinal(d3.schemeTableau10)
             .domain([...new Set(values)]);
  }
  if (metric === 'Residual') {
    const maxAbs = d3.max(values, d => Math.abs(d)) || 1;
    return d3.scaleSequential(d3.interpolateRdBu).domain([-maxAbs, maxAbs]);
  }
  const extent = d3.extent(values);
  return d3.scaleSequential(d3.interpolateViridis).domain(extent);
}

function updateMap() {

  // remove previous markers
  layerGroup.clearLayers();

  // determine coloring metric
  const metric = document.getElementById("metric").value;

  const values = state.houses
    .map(d => Number(d.metric))
    .filter(v => !isNaN(v));

  const color = getColorScale(metric, values);
  updateLegend(metric, color, values);
  const sizeScale = d3.scaleSqrt()
                        .domain(d3.extent(values))
                        .range([4,12]);

  state.houses.forEach(d => {

    const lat = Number(d.Latitude);
    const lon = Number(d.Longitude);
    const val = Number(d.metric);

    const markerColor = isNaN(val) ? "blue" : color(val);

    const radius = isNaN(val) ? 5: sizeScale(Math.abs(val));

    const marker = L.circleMarker([lat,lon],{
      //icon: blueIcon
      radius: radius,
      fillColor: markerColor,
      color: "#333",
      weight: 1,
      fillOpacity: 0.8
    });

    //marker.bindPopup(`
     // <strong>${d.Neighborhood}</strong><br>
     // Sale Price: ${currency(d.Sale_Price)}<br>
     // Predicted: ${currency(d.Predicted_Price)}<br>
     // Residual: ${currency(d.Residual)}<br>
     // Living Area: ${d.Gr_Liv_Area}<br>
     // Overall Quality: ${d.Overall_Cond}<br>
     // Year Built: ${d.Year_Built}
    //`);

       marker.bindTooltip(`
      <strong>${d.Neighborhood}</strong><br>
      Sale Price: ${currency(d.Sale_Price)}<br>
      Predicted: ${currency(d.Predicted_Price)}<br>
      Residual: ${currency(d.Residual)}<br>
      Interval: ${currency(d.PI_Lower)} - ${currency(d.PI_Upper)}<br>
      Living Area: ${d.Gr_Liv_Area}<br>
      Overall Quality: ${d.Overall_Cond}<br>
      Year Built: ${d.Year_Built}
    `,{
      direction:"top",
      offset:[0,-10]
    });

    marker.addTo(layerGroup);

    // attach tooltip events after marker is added
    marker.on("add", function () {

      const el = marker.getElement();

      d3.select(el)
        .on("mouseover", (event) => {

          const tip = `
            <strong>${d.Neighborhood}</strong><br>
            Price: ${currency(d.Sale_Price)}<br>
            Area: ${d3.format(',')(d.Gr_Liv_Area)} sqft<br>
            Quality: ${d3.format('.2f')(d.Overall_Cond)}<br>
            Year Built: ${d.Year_Built}
          `;

          showTip(event, tip);

        })
        .on("mouse leave", hideTip);

    });

  });

  console.log("Total markers:", layerGroup.getLayers().length);

  // auto zoom to visible houses
  if (layerGroup.getLayers().length > 0) {
    map.fitBounds(layerGroup.getBounds());
  }
}

let legend = L.control({position: "bottomright"});

legend.onAdd = function(){
  this._div = L.DomUtil.create('div', 'info legend');
  return this._div;
}
legend.addTo(map);

function updateLegend(metric, colorScale, values) {
  const div = legend._div;

  if (!values.length) {
    div.innerHTML = "";
    return;
  }

  // Cluster (categorical)
  if (metric === "Cluster") {
    const unique = [...new Set(values)].sort();
    div.innerHTML = "<strong>Cluster</strong><br>";
    
    unique.forEach(v => {
      div.innerHTML += `
        <div>
          <span style="
            display:inline-block;
            width:12px;
            height:12px;
            background:${colorScale(v)};
            margin-right:6px;
          "></span>
          ${v}
        </div>
      `;
    });

    return;
  }

  // Continuous (price, residual, etc.)
  const min = d3.min(values);
  const max = d3.max(values);

  const steps = 6;
  const stepSize = (max - min) / steps;

  div.innerHTML = `<strong>${metric}</strong><br>`;

  for (let i = 0; i < steps; i++) {
    const v1 = min + i * stepSize;
    const v2 = v1 + stepSize;

    div.innerHTML += `
      <div>
        <span style="
          display:inline-block;
          width:12px;
          height:12px;
          background:${colorScale((v1 + v2)/2)};
          margin-right:6px;
        "></span>
        ${currency(v1)} – ${currency(v2)}
      </div>
    `;
  }
}


function baseSvg(selector) {
  const svg = d3.select(selector);
  svg.selectAll('*').remove();
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  const margin = { top: 20, right: 20, bottom: 65, left: 70 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  return { svg, g, width, height, innerW, innerH, margin };
}

function showTip(event, html) {
  tooltip.style('opacity', 1).html(html)
    .style('left', `${event.pageX + 6}px`)
    .style('top', `${event.pageY - 14}px`);
}
function hideTip() { tooltip.style('opacity', 0); }

async function refresh() {
  console.log("REFRESH TRIGGERED");
  const q = buildQuery();
  console.log("QUERY STRING:", q);
  const suffix = q ? `?${q}` : '';
  state.houses = await getJSON(`/api/houses${suffix}`);
  drawMetricCards();
  updateMap();
}

async function init() {

  state.summary = await getJSON('/api/summary');

  initControls(state.summary);

  // Attach event listeners AFTER page is ready

  document.getElementById('applyBtn').addEventListener('click', refresh);
  document.getElementById('metric').addEventListener('change', refresh);

  document.getElementById('resetBtn').addEventListener('click', async () => {
    document.getElementById('neighborhood').value = 'All';
    document.getElementById('metric').value = 'Sale_Price';
    document.getElementById('minPrice').value = '';
    document.getElementById('maxPrice').value = '';
    document.getElementById('minYear').placeholder = state.summary.year_min;
    document.getElementById('maxYear').value = state.summary.year_max;
    document.getElementById('minQual').value = state.summary.quality_min;
    await refresh();
  });

  await refresh();
}

window.addEventListener('resize', () => {
  map.invalidateSize();
});

init();

// ── Custom Prediction Form ────────────────────────────────────────

function drawWaterfall(baseValue, shapValues, predictedPrice) {
  const svg = d3.select('#shap-waterfall');
  svg.selectAll('*').remove();

  const W = svg.node().clientWidth || 800;
  const H = svg.node().clientHeight || 320;
  const margin = { top: 18, right: 110, bottom: 40, left: 145 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // Build waterfall rows: base → each shap feature → predicted (total)
  const rows = [];
  let running = baseValue;
  rows.push({ label: 'Base Price', start: 0, end: baseValue, value: baseValue, isBase: true });
  shapValues.forEach(d => {
    const start = running;
    const end = running + d.value;
    rows.push({ label: d.feature, start, end, value: d.value, isBase: false });
    running = end;
  });
  rows.push({ label: 'Predicted Price', start: 0, end: predictedPrice, value: predictedPrice, isTotal: true });

  const allVals = rows.flatMap(r => [r.start, r.end]);
  const xMin = d3.min(allVals);
  const xMax = d3.max(allVals);
  const xPad = (xMax - xMin) * 0.02;

  const x = d3.scaleLinear().domain([xMin - xPad, xMax + xPad]).range([0, innerW]);
  const y = d3.scaleBand().domain(rows.map(r => r.label)).range([0, innerH]).padding(0.25);

  const barH = y.bandwidth();

  // Gridlines
  g.append('g')
    .attr('class', 'grid')
    .call(d3.axisBottom(x).ticks(5).tickSize(innerH).tickFormat(''))
    .call(gg => gg.select('.domain').remove())
    .call(gg => gg.selectAll('line').attr('stroke', '#e2e8f0'))
    .attr('transform', 'translate(0,0)');

  // Zero line
  g.append('line')
    .attr('x1', x(0)).attr('x2', x(0))
    .attr('y1', 0).attr('y2', innerH)
    .attr('stroke', '#94a3b8').attr('stroke-width', 1).attr('stroke-dasharray', '4,3');

  // Bars
  rows.forEach(r => {
    let fill, barX, barW;
    if (r.isBase || r.isTotal) {
      fill = r.isTotal ? '#304ffe' : '#64748b';
      barX = x(0);
      barW = Math.abs(x(r.end) - x(0));
    } else {
      fill = r.value >= 0 ? '#22c55e' : '#ef4444';
      barX = r.value >= 0 ? x(r.start) : x(r.end);
      barW = Math.abs(x(r.end) - x(r.start));
    }

    g.append('rect')
      .attr('x', barX).attr('y', y(r.label))
      .attr('width', Math.max(barW, 1)).attr('height', barH)
      .attr('fill', fill).attr('rx', 3);

    // Value label at end of bar
    const labelX = r.isBase || r.isTotal
      ? x(r.end) + 5
      : (r.value >= 0 ? x(r.end) + 5 : x(r.end) - 5);
    const anchor = (r.isBase || r.isTotal || r.value >= 0) ? 'start' : 'end';
    const sign = (!r.isBase && !r.isTotal && r.value >= 0) ? '+' : '';

    g.append('text')
      .attr('x', labelX).attr('y', y(r.label) + barH / 2 + 4)
      .attr('text-anchor', anchor)
      .attr('font-size', '0.75rem')
      .attr('fill', '#1e293b')
      .text(sign + currency(r.isBase || r.isTotal ? r.end : r.value));
  });

  // Y axis (feature labels)
  g.append('g')
    .call(d3.axisLeft(y).tickSize(0))
    .call(gg => gg.select('.domain').remove())
    .selectAll('text')
    .attr('font-size', '0.78rem')
    .attr('fill', d => {
      const r = rows.find(r => r.label === d);
      return r?.isTotal ? '#304ffe' : '#334155';
    })
    .attr('font-weight', d => {
      const r = rows.find(r => r.label === d);
      return (r?.isBase || r?.isTotal) ? '600' : '400';
    });

  // X axis
  g.append('g')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d => `$${d3.format(',.0f')(d)}`))
    .call(gg => gg.select('.domain').attr('stroke', '#cbd5e1'))
    .selectAll('text').attr('font-size', '0.75rem');
}

async function submitPrediction() {
  const btn = document.getElementById('predictBtn');
  btn.textContent = 'Predicting…';
  btn.disabled = true;

  const body = {
    overall_cond:   document.getElementById('pred-overall-cond').value,
    gr_liv_area:    parseFloat(document.getElementById('pred-gr-liv-area').value) || null,
    neighborhood:   document.getElementById('pred-neighborhood').value,
    year_built:     parseFloat(document.getElementById('pred-year-built').value) || null,
    total_bsmt_sf:  parseFloat(document.getElementById('pred-total-bsmt-sf').value) || null,
    garage_cars:    parseFloat(document.getElementById('pred-garage-cars').value),
    full_bath:      parseFloat(document.getElementById('pred-full-bath').value),
    bedroom_abvgr:  parseFloat(document.getElementById('pred-bedroom-abvgr').value),
    house_style:    document.getElementById('pred-house-style').value,
    central_air:    document.getElementById('pred-central-air').value,
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error('Prediction request failed');
    const data = await res.json();

    document.getElementById('pred-price').textContent = currency(data.predicted_price);
    document.getElementById('pred-interval').textContent =
      `${currency(data.lower)}  –  ${currency(data.upper)}`;

    const result = document.getElementById('pred-result');
    result.classList.remove('hidden');

    drawWaterfall(data.shap_base_value, data.shap_values, data.predicted_price);
  } catch (err) {
    console.error(err);
    alert('Prediction failed. Check the console for details.');
  } finally {
    btn.textContent = 'Predict Price';
    btn.disabled = false;
  }
}

document.getElementById('predictBtn').addEventListener('click', submitPrediction);
