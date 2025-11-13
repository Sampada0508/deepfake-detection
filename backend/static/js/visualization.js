// visualization.js - safer, smaller charts, no auto-refresh, supports backend key "recent" or "history"
const API = "/api/visualization-data";

let trendChart = null, pieChart = null, barChart = null;

async function fetchViz() {
  try {
    const res = await fetch(API);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API ${res.status}: ${txt}`);
    }
    const payload = await res.json();
    if (payload.status && payload.status !== "ok") {
      throw new Error(payload.message || JSON.stringify(payload));
    }
    // if backend wraps under {status:ok, data:...}
    return payload.data ? payload.data : payload;
  } catch (err) {
    console.error("fetchViz error:", err);
    alert("Visualization load failed: " + err.message);
    return null;
  }
}

function safeArray(a, len = 0) {
  if (!Array.isArray(a)) return Array(len).fill(0);
  return a;
}

function makeChartOptions() {
  return {
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#cfeff6' } } },
    scales: {
      x: { ticks: { color: '#9fb8c3' } },
      y: { ticks: { color: '#9fb8c3', beginAtZero: true } }
    }
  };
}

function drawLine(labels, fakeData, realData) {
  const ctx = document.getElementById('trendChart').getContext('2d');
  const cfg = {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Fake', data: fakeData, fill: true, tension: 0.3, borderColor: 'rgba(255,99,132,0.9)', backgroundColor: 'rgba(255,99,132,0.08)', pointRadius: 2 },
        { label: 'Real', data: realData, fill: true, tension: 0.3, borderColor: 'rgba(75,192,192,0.9)', backgroundColor: 'rgba(75,192,192,0.08)', pointRadius: 2 }
      ]
    },
    options: makeChartOptions()
  };
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, cfg);
}

function drawPie(bins) {
  const ctx = document.getElementById('pieChart').getContext('2d');
  const labels = Object.keys(bins || {});
  const data = Object.values(bins || {});
  const cfg = {
    type: 'pie',
    data: { labels, datasets: [{ data, backgroundColor: ['#ffd1d1','#ffdca6','#fff59b','#b1f5d0','#9fe9ff'] }] },
    options: { maintainAspectRatio: false, plugins: { legend: { labels: { color: '#cfeff6' } } } }
  };
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(ctx, cfg);
}

function drawBar(labels, totals) {
  const ctx = document.getElementById('barChart').getContext('2d');
  const cfg = {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Total', data: totals, borderRadius: 6 }] },
    options: makeChartOptions()
  };
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx, cfg);
}

/* HISTORY TABLE */
let history = [];
const PAGE_SIZE = 8;
let page = 1;
let currentFilter = 'all';

function formatConf(v) {
  if (v === undefined || v === null) return '-';
  return Number(v).toFixed(2);
}

function renderTable() {
  const tbody = document.querySelector('#historyTable tbody');
  tbody.innerHTML = '';
  const filtered = history.filter(r => {
    if (currentFilter === 'all') return true;
    if (['image','video','error'].includes(currentFilter)) return r.type === currentFilter;
    if (['fake','real'].includes(currentFilter)) return r.label === currentFilter;
    return true;
  });
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  if (page > totalPages) page = totalPages;
  const start = (page - 1) * PAGE_SIZE;
  const slice = filtered.slice(start, start + PAGE_SIZE);
  slice.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${start + idx + 1}</td>
      <td>${r.file || '-'}</td>
      <td>${r.type || '-'}</td>
      <td class="${r.label==='fake' ? 'label-fake' : r.label==='real' ? 'label-real' : 'label-error'}">${r.label || '-'}</td>
      <td>${formatConf(r.confidence)}</td>
      <td>${r.time || '-'}</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('pageInfo').textContent = `Page ${page} / ${totalPages}`;
  document.getElementById('prevPage').disabled = page <= 1;
  document.getElementById('nextPage').disabled = page >= totalPages;
}

function setupControls() {
  document.getElementById('prevPage').addEventListener('click', () => { if (page>1) { page--; renderTable(); }});
  document.getElementById('nextPage').addEventListener('click', () => { page++; renderTable(); });
  document.getElementById('filterType').addEventListener('change', (e) => { currentFilter = e.target.value; page = 1; renderTable(); });
  document.getElementById('downloadCsv').addEventListener('click', downloadCSV);
  document.getElementById('refreshBtn').addEventListener('click', refreshAll);
}

function downloadCSV() {
  const filtered = history;
  const rows = [['id','file','type','label','confidence','time']];
  filtered.forEach((r,i) => rows.push([i+1, r.file || '', r.type || '', r.label || '', r.confidence || '', r.time || '']));
  const csv = rows.map(r => r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `truthlens_history_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* MAIN */
async function refreshAll() {
  const data = await fetchViz();
  if (!data) return;

  // tolerate both data.recent and data.history
  const recent = Array.isArray(data.recent) ? data.recent : (Array.isArray(data.history) ? data.history : []);
  const dates = Array.isArray(data.dates) ? data.dates : [];
  const fake_counts = Array.isArray(data.fake_counts) ? data.fake_counts : dates.map(_=>0);
  const real_counts = Array.isArray(data.real_counts) ? data.real_counts : dates.map(_=>0);
  const confidence_bins = data.confidence_bins || {};

  // ensure arrays have same length
  const L = dates.length || Math.max(fake_counts.length, real_counts.length);
  const safeDates = dates.length ? dates : Array.from({length:L}, (_,i)=>`Day ${i+1}`);

  drawLine(safeDates, fake_counts, real_counts);
  const totals = safeDates.map((_,i) => ( (fake_counts[i]||0) + (real_counts[i]||0) ));
  drawBar(safeDates, totals);
  drawPie(confidence_bins);

  history = recent.map((r, idx) => ({
    id: r.id || idx+1,
    file: r.file || r.name || r.filename || '-',
    type: r.type || (r.file && r.file.toLowerCase().endsWith('.mp4') ? 'video' : 'image'),
    label: r.label || (r.error ? 'error' : '-'),
    confidence: r.confidence !== undefined ? r.confidence : (r.conf ? r.conf : 0),
    time: r.time || r.datetime || r.date || (new Date()).toLocaleString()
  }));

  page = 1;
  renderTable();
}

document.addEventListener('DOMContentLoaded', () => {
  setupControls();
  refreshAll();
});