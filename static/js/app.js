/* StreetGuard AI — app.js  |  4-step workflow */

// ── refs ──────────────────────────────────────────────────────────────────────
const dz         = document.getElementById('dz');
const fileInput  = document.getElementById('fileInput');
const browseBtn  = document.getElementById('browseBtn');
const dzEmpty    = document.getElementById('dzEmpty');
const dzFilled   = document.getElementById('dzFilled');
const previewImg = document.getElementById('previewImg');
const clearBtn   = document.getElementById('clearBtn');
const analyseBtn = document.getElementById('analyseBtn');
const btnTxt     = document.getElementById('btnTxt');
const btnLoad    = document.getElementById('btnLoad');

const results    = document.getElementById('results');
const resetBtn   = document.getElementById('resetBtn');

// Step 2
const statusEmoji= document.getElementById('statusEmoji');
const statusBig  = document.getElementById('statusBig');
const confBadge  = document.getElementById('confBadge');
const scannedImg = document.getElementById('scannedImg');
const scanTag    = document.getElementById('scanTag');
const confPct    = document.getElementById('confPct');
const confBar    = document.getElementById('confBar');
const probRows   = document.getElementById('probRows');
const healthyBox = document.getElementById('healthyBox');

// Step 3
const s3          = document.getElementById('s3');
const urgencyAlert= document.getElementById('urgencyAlert');
const urgencyIcon = document.getElementById('urgencyIcon');
const urgencyTxt  = document.getElementById('urgencyTxt');
const dcName      = document.getElementById('dcName');
const diffBars    = document.getElementById('diffBars');
const flagsDiv    = document.getElementById('flags');
const tabSymptoms = document.getElementById('tab-symptoms');
const tabCure     = document.getElementById('tab-cure');
const tabFirstaid = document.getElementById('tab-firstaid');

// Step 4
const gpsBtn     = document.getElementById('gpsBtn');
const citySelect = document.getElementById('citySelect');
const ngoResult  = document.getElementById('ngoResult');
const ngoCityLbl = document.getElementById('ngoCityLabel');
const ngoList    = document.getElementById('ngoList');
const ngoLoading = document.getElementById('ngoLoading');
const ngoError   = document.getElementById('ngoError');

let selectedFile = null;

// ── File handling ──────────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dz.addEventListener('click', e => { if (e.target !== clearBtn) fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', ()  => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});

function loadFile(f) {
  selectedFile = f;
  previewImg.src = URL.createObjectURL(f);
  dzEmpty.classList.add('hidden');
  dzFilled.classList.remove('hidden');
  analyseBtn.disabled = false;
}

clearBtn.addEventListener('click', e => { e.stopPropagation(); resetUpload(); });
function resetUpload() {
  selectedFile = null; previewImg.src = ''; fileInput.value = '';
  dzEmpty.classList.remove('hidden'); dzFilled.classList.add('hidden');
  analyseBtn.disabled = true;
}

// ── Analyse ───────────────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  setLoading(true);

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch(e) {
    alert('Network error — is Flask running?');
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  btnTxt.classList.toggle('hidden', on);
  btnLoad.classList.toggle('hidden', !on);
  analyseBtn.disabled = on;
}

// ── Render all results ────────────────────────────────────────────────────────
function renderResults(data) {
  const infected = data.label === 'infected';
  const cls      = infected ? 'infected' : 'healthy';
  const emoji    = infected ? '⚠️' : '✅';
  const tag      = infected ? '⚠ Infected' : '✓ Healthy';

  // Step 2
  statusEmoji.textContent = emoji;
  statusBig.textContent   = data.label.charAt(0).toUpperCase() + data.label.slice(1);
  statusBig.className     = 'status-big ' + cls;
  confBadge.textContent   = data.confidence + '%';
  scannedImg.src          = data.image_url;
  scanTag.textContent     = tag;
  scanTag.className       = 'scan-tag ' + cls;
  confPct.textContent     = data.confidence + '%';
  confBar.className       = 'bar-fill ' + cls;

  // Prob rows
  probRows.innerHTML = '';
  for (const [name, pct] of Object.entries(data.probabilities)) {
    const rowCls = name === 'healthy' ? 'healthy' : 'infected';
    probRows.innerHTML += `
      <div class="prob-row">
        <span class="prob-name">${name}</span>
        <div class="prob-track"><div class="prob-fill ${rowCls}" style="width:0" data-w="${pct}"></div></div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  }

  healthyBox.classList.toggle('hidden', infected);

  // Step 3
  if (infected && data.disease) {
    s3.classList.remove('hidden');
    renderDisease(data.disease);
  } else {
    s3.classList.add('hidden');
  }

  results.classList.remove('hidden');
  resetBtn.classList.remove('hidden');

  // Animate bars
  requestAnimationFrame(() => setTimeout(() => {
    confBar.style.width = data.confidence + '%';
    document.querySelectorAll('.prob-fill').forEach(el => el.style.width = el.dataset.w + '%');
  }, 80));

  // Scroll to results
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Render disease ────────────────────────────────────────────────────────────
function renderDisease(d) {
  // Urgency
  urgencyAlert.className = 'urgency-alert ' + d.urgency;
  const icons = { critical: '🚨', high: '⚠️', medium: '⚡', low: 'ℹ️' };
  urgencyIcon.textContent = icons[d.urgency] || '⚠️';
  urgencyTxt.textContent  = d.urgency_label;

  // Disease name
  dcName.textContent = d.display_name;

  // Differential bars
  diffBars.innerHTML = '';
  (d.top3_diseases || []).forEach((item, i) => {
    diffBars.innerHTML += `
      <div class="diff-bar-row">
        <span class="diff-name">${item.name}</span>
        <div class="diff-track"><div class="diff-fill ${i === 0 ? 'top' : 'other'}" style="width:0" data-w="${item.pct}"></div></div>
        <span class="diff-pct">${item.pct}%</span>
      </div>`;
  });

  // Flags
  flagsDiv.innerHTML = '';
  if (d.is_contagious)
    flagsDiv.innerHTML += `<span class="flag flag-red">⚠ Contagious to other dogs</span>`;
  if (d.zoonotic)
    flagsDiv.innerHTML += `<span class="flag flag-yellow">⚠ Can spread to humans</span>`;
  if (!d.is_contagious && !d.zoonotic)
    flagsDiv.innerHTML += `<span class="flag flag-green">✓ Not contagious</span>`;

  // Tab content
  tabSymptoms.innerHTML = (d.symptoms || []).map(s =>
    `<div class="tab-item"><span class="dot-warn"></span><span>${s}</span></div>`).join('');

  tabCure.innerHTML = (d.basic_cure || []).map((c, i) =>
    `<div class="tab-item"><span class="num">${i+1}</span><span>${c}</span></div>`).join('');

  tabFirstaid.innerHTML = (d.home_first_aid || []).map(a =>
    `<div class="tab-item"><span class="dot-aid"></span><span>${a}</span></div>`).join('');

  // Animate diff bars
  requestAnimationFrame(() => setTimeout(() => {
    document.querySelectorAll('.diff-fill').forEach(el => el.style.width = el.dataset.w + '%');
  }, 200));
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
  });
});

// ── NGO / GPS ─────────────────────────────────────────────────────────────────
gpsBtn.addEventListener('click', () => {
  if (!navigator.geolocation) {
    showNgoError('Geolocation not supported by your browser.');
    return;
  }
  setNgoLoading(true);
  navigator.geolocation.getCurrentPosition(
    pos => fetchNgo({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
    ()  => { setNgoLoading(false); showNgoError('Location permission denied. Please select a city below.'); }
  );
});

citySelect.addEventListener('change', () => {
  if (citySelect.value) fetchNgo({ city: citySelect.value });
});

function fetchNgo(payload) {
  setNgoLoading(true);
  fetch('/ngo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(data => {
    setNgoLoading(false);
    if (data.error) { showNgoError(data.error); return; }
    renderNgo(data);
  })
  .catch(() => { setNgoLoading(false); showNgoError('Failed to fetch contacts.'); });
}

function renderNgo(data) {
  const distTxt = data.distance_km != null ? ` (≈${data.distance_km} km away)` : '';
  ngoCityLbl.textContent = `📍 Showing contacts for: ${data.city}${distTxt}`;

  ngoList.innerHTML = '';
  if (!data.contacts || data.contacts.length === 0) {
    ngoList.innerHTML = '<p style="color:var(--muted2);font-size:.85rem">No specific contacts found for this location.</p>';
  } else {
    data.contacts.forEach(c => {
      const icon      = c.type === 'clinic' ? '🏥' : '🐾';
      const emClass   = c.emergency ? 'emergency' : '';
      const emBadge   = c.emergency ? '<span class="em-badge">🚨 Emergency</span>' : '';
      ngoList.innerHTML += `
        <div class="ngo-card ${emClass}">
          <div class="ngo-icon">${icon}</div>
          <div style="flex:1;min-width:0">
            <div class="ngo-name">${c.name}</div>
            <div class="ngo-addr">📍 ${c.address}</div>
            <div class="ngo-meta">
              <a class="ngo-phone" href="tel:${c.phone}">📞 ${c.phone}</a>
              <span class="ngo-type-badge ${c.type}">${c.type === 'clinic' ? 'Vet Clinic' : 'NGO'}</span>
              ${emBadge}
            </div>
          </div>
        </div>`;
    });
  }

  ngoResult.classList.remove('hidden');
  ngoError.classList.add('hidden');
}

function setNgoLoading(on) {
  ngoLoading.classList.toggle('hidden', !on);
  if (on) { ngoResult.classList.add('hidden'); ngoError.classList.add('hidden'); }
}
function showNgoError(msg) {
  ngoError.textContent = msg;
  ngoError.classList.remove('hidden');
  ngoResult.classList.add('hidden');
}

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  results.classList.add('hidden');
  resetBtn.classList.add('hidden');
  s3.classList.add('hidden');
  ngoResult.classList.add('hidden');
  ngoError.classList.add('hidden');
  citySelect.value = '';
  resetUpload();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});