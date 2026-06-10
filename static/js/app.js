document.addEventListener('DOMContentLoaded', () => {

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {

    const target = this.getAttribute('href');

    if (target === '#') return;

    e.preventDefault();

    const section = document.querySelector(target);

    if (section) {
      section.scrollIntoView({
        behavior: 'smooth'
      });
    }
  });
});

  // ── "Scan a Dog" / "Analyse Now" / "Upload a Photo" → trigger file picker ──
  const galleryInput = createImageInput(false);
  const cameraInput = createImageInput(true);
  window.openDogScanOptions = openDogScanOptions;

  function createImageInput(useCamera) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    if (useCamera) input.setAttribute('capture', 'environment');
    input.style.display = 'none';
    document.body.appendChild(input);
    return input;
  }

  function isTouchPhone() {
    return window.matchMedia('(max-width: 768px), (pointer: coarse)').matches;
  }

  async function readApiJson(response, fallbackMessage) {
    const raw = await response.text();
    const contentType = response.headers.get('content-type') || '';

    if (!raw.trim()) {
      throw new Error(`${fallbackMessage} Empty response from server.`);
    }

    if (!contentType.includes('application/json')) {
      throw new Error(`${fallbackMessage} Server returned ${response.status || 'an invalid'} response.`);
    }

    try {
      return JSON.parse(raw);
    } catch (err) {
      throw new Error(`${fallbackMessage} Server returned invalid JSON.`);
    }
  }

  function openDogScanOptions() {
    if (!isTouchPhone()) {
      galleryInput.click();
      return;
    }

    const existing = document.querySelector('.scan-choice-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.className = 'scan-choice-overlay';
    overlay.innerHTML = `
      <div class="scan-choice-sheet" role="dialog" aria-modal="true" aria-label="Choose image source">
        <button class="scan-choice-btn" type="button" data-source="camera">Take Photo</button>
        <button class="scan-choice-btn" type="button" data-source="gallery">Upload Photo</button>
        <button class="scan-choice-cancel" type="button">Cancel</button>
      </div>`;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.addEventListener('click', event => {
      if (event.target === overlay || event.target.classList.contains('scan-choice-cancel')) {
        close();
        return;
      }

      const source = event.target.dataset.source;
      if (source === 'camera') {
        close();
        cameraInput.click();
      } else if (source === 'gallery') {
        close();
        galleryInput.click();
      }
    });
  }

  const scanTriggers = document.querySelectorAll(
    '.nav-cta, .btn-primary, .btn-outline, .btn-white'
  );

  scanTriggers.forEach(btn => {
    const label = btn.textContent.trim().toLowerCase();
    if (
      label.includes('scan') ||
      label.includes('analys') ||
      label.includes('upload') ||
      label.includes('start')
    ) {
      btn.addEventListener('click', openDogScanOptions);
    }
  });

  // ── File selected → POST to /predict ──
  galleryInput.addEventListener('change', handleDogImageSelected);
  cameraInput.addEventListener('change', handleDogImageSelected);

  async function handleDogImageSelected(event) {
    const input = event.target;
    const file = input.files[0];
    if (!file) return;

    window.lastUploadedDogImage = URL.createObjectURL(file);

    // Show loading state in report card
    const reportSection = document.getElementById('report-card');
    const reportContent = document.getElementById('report-content');
    if (reportSection) {
      reportSection.style.display = 'block';
      reportContent.innerHTML = `
        <div class="report-loading">
          <div class="spinner"></div>
          <p>Analysing dog image…</p>
        </div>`;
      reportSection.scrollIntoView({ behavior: 'smooth' });
    }

    try {
      const uploadFile = await prepareImageForUpload(file);
      const formData = new FormData();
      formData.append('file', uploadFile, uploadFile.name || 'dog-photo.jpg');

      // FIX 1: AbortController timeout — mobile networks can be slow.
      // Without this the fetch hangs forever on a bad connection.
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort(), 60000); // 60 s

      let response;
      try {
        response = await fetch('/predict', {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
      } catch (fetchErr) {
        clearTimeout(timeoutId);
        // AbortError = our timeout; TypeError = network gone
        const msg = fetchErr.name === 'AbortError'
          ? 'Request timed out. Please check your connection and try again.'
          : 'Network error — make sure your device can reach the server.';
        if (reportContent) {
          reportContent.innerHTML = `<div class="report-error">Error: ${msg}</div>`;
        }
        showToast('Error: ' + msg);
        input.value = '';
        return;
      }
      clearTimeout(timeoutId);

      let data = {};
      try {
        data = await readApiJson(response, 'Could not read analysis result.');
      } catch (err) {
        let msg = err.message || 'Could not read analysis result.';
        if (response.status === 413) msg = 'Image is too large. Please try a smaller photo.';
        else if (response.status === 500) msg = 'Server error during analysis. Try again.';
        else if (response.status === 400) msg = 'Invalid image file. Please try a different photo.';
        if (reportContent) {
          reportContent.innerHTML = `<div class="report-error">Error: ${msg}</div>`;
        }
        showToast('Error: ' + msg);
        input.value = '';
        return;
      }

      if (!response.ok || data.error) {
        const message = data.error || 'Analysis failed. Please try another image.';
        if (reportContent) {
          reportContent.innerHTML = `<div class="report-error">Error: ${message}</div>`;
        }
        showToast('Error: ' + message);
        input.value = '';
        return;
      }

      showReportCard(data);

    } catch (err) {
      console.error(err);
      if (reportContent) {
        reportContent.innerHTML = `<div class="report-error">Analysis failed. Please try again.</div>`;
      }
      showToast('Analysis failed — please try again');
    }

    input.value = '';
  }

  // FIX 3: prepareImageForUpload — also resize HEIC/HEIF on mobile by
  // converting them through a canvas after the browser decodes them.
  // Previously HEIC files were sent raw (potentially 10–15 MB) and Flask
  // would time out or the mobile network would drop the upload.
  function prepareImageForUpload(file) {
    const MAX_SIDE = 1200;
    const MAX_BYTES = 3 * 1024 * 1024; // 3 MB threshold before we bother resizing

    // Always convert HEIC/HEIF via canvas — mobile Safari can decode them
    // but the raw file bytes are huge and unsupported by many servers.
    const isHEIC = /heic|heif/i.test(file.type) || /heic|heif/i.test(file.name);

    if (!isHEIC && file.size <= MAX_BYTES) {
      return Promise.resolve(file);
    }

    return new Promise(resolve => {
      const img = new window.Image();
      const objectUrl = URL.createObjectURL(file);

      img.onload = () => {
        URL.revokeObjectURL(objectUrl);
        const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
        const canvas = document.createElement('canvas');
        canvas.width  = Math.max(1, Math.round(img.width  * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          if (!blob) { resolve(file); return; }
          resolve(new File([blob], 'dog-photo.jpg', { type: 'image/jpeg' }));
        }, 'image/jpeg', 0.85);
      };

      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        resolve(file); // best effort fallback
      };

      img.src = objectUrl;
    });
  }

  // ── "Contact Us" button ──
  document.querySelectorAll('.btn-pink').forEach(btn => {
    if (btn.textContent.trim().toLowerCase().includes('contact')) {
      btn.addEventListener('click', () => {
        showToast('Contact Details: Email: Kanmanijayakumar26022007@gmail.com phone number: 7904163848');
      });
    }
  });

  // ── Sticky nav shadow on scroll ──
  const nav = document.querySelector('nav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.style.boxShadow = window.scrollY > 10
        ? '0 4px 30px rgba(190,24,93,0.13)'
        : '0 2px 20px rgba(190,24,93,0.06)';
    }, { passive: true });
  }

  // ── Fade-in on scroll ──
  const fadeEls = document.querySelectorAll(
    '.card, .about-content, .about-img, .cta-block, .stat'
  );

  fadeEls.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(18px)';
    el.style.transition = 'opacity 0.55s ease, transform 0.55s ease';
  });

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  fadeEls.forEach(el => observer.observe(el));

  // ── NGO section wiring ──
  const ngoSearchBtn  = document.getElementById('ngo-search-btn');
  const ngoCityInput  = document.getElementById('ngo-city-input');
  const ngoGpsBtn     = document.getElementById('ngo-gps-btn');
  const ngoResults    = document.getElementById('ngo-results');

  if (ngoSearchBtn) {
    ngoSearchBtn.addEventListener('click', () => {
      const city = ngoCityInput ? ngoCityInput.value.trim() : '';
      if (!city) { showToast('Please enter a city name'); return; }
      fetchNGO({ city });
    });
  }

  if (ngoCityInput) {
    ngoCityInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') ngoSearchBtn && ngoSearchBtn.click();
    });
  }

  if (ngoGpsBtn) {
    ngoGpsBtn.addEventListener('click', () => {
      if (!navigator.geolocation) {
        showToast('Geolocation not supported by your browser');
        return;
      }
      if (!window.isSecureContext) {
        showToast('Location needs HTTPS on mobile. Please search by city instead.');
        return;
      }
      ngoGpsBtn.textContent = 'Locating…';
      ngoGpsBtn.disabled = true;

      // FIX 4: geolocation options — mobile browsers often fail silently
      // without a timeout. enableHighAccuracy can also block forever on mobile.
      const geoOptions = {
        enableHighAccuracy: false,  // false = faster on mobile (uses cell/wifi)
        timeout: 10000,             // 10 s — surface error instead of hanging
        maximumAge: 60000,          // accept a cached position up to 1 min old
      };

      navigator.geolocation.getCurrentPosition(
        pos => {
          ngoGpsBtn.textContent = 'Use My Location';
          ngoGpsBtn.disabled = false;
          fetchNGO({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        },
        err => {
          ngoGpsBtn.textContent = 'Use My Location';
          ngoGpsBtn.disabled = false;
          // FIX 5: map the numeric error code to a readable message
          const geoErrors = {
            1: 'Location access denied. Please allow location in browser settings.',
            2: 'Location unavailable. Try entering your city name instead.',
            3: 'Location timed out. Try entering your city name instead.',
          };
          showToast(geoErrors[err.code] || 'Could not get location.');
        },
        geoOptions
      );
    });
  }

  // ── Toast helper ──
  function showToast(message) {
    const existing = document.querySelector('.ph-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'ph-toast';
    toast.textContent = message;
    Object.assign(toast.style, {
      position: 'fixed',
      bottom: '2rem',
      left: '50%',
      transform: 'translateX(-50%) translateY(20px)',
      background: '#831843',
      color: '#fce7f3',
      padding: '.75rem 1.5rem',
      borderRadius: '8px',
      fontSize: '.85rem',
      fontFamily: "'DM Sans', sans-serif",
      fontWeight: '500',
      boxShadow: '0 8px 30px rgba(131,24,67,0.35)',
      zIndex: '9999',
      opacity: '0',
      transition: 'opacity .3s ease, transform .3s ease',
      // FIX 6: allow toast text to wrap on narrow mobile screens
      whiteSpace: 'normal',
      maxWidth: '90vw',
      textAlign: 'center',
    });

    document.body.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateX(-50%) translateY(0)';
    });
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(-50%) translateY(20px)';
      setTimeout(() => toast.remove(), 350);
    }, 3200);
  }


const hamburger = document.querySelector('.nav-hamburger');
const drawer    = document.querySelector('.nav-drawer');
const overlay   = document.querySelector('.nav-drawer-overlay');
hamburger?.addEventListener('click', () => {
  drawer.classList.toggle('open');
  overlay.style.display = drawer.classList.contains('open') ? 'block' : 'none';
});
overlay?.addEventListener('click', () => {
  drawer.classList.remove('open');
  overlay.style.display = 'none';
});
document.querySelectorAll('.nav-drawer a').forEach(link => {
  link.addEventListener('click', () => {
    drawer.classList.remove('open');
    overlay.style.display = 'none';
  });
});
const closeBtn = document.querySelector('.drawer-close');

closeBtn?.addEventListener('click', () => {
  drawer.classList.remove('open');
  overlay.style.display = 'none';
});

  // ── Fetch NGO data from backend ──
  async function fetchNGO(params) {
    if (!ngoResults) return;
    ngoResults.innerHTML = `
      <div class="report-loading">
        <div class="spinner"></div>
        <p>Finding nearby vets &amp; NGOs…</p>
      </div>`;

    // FIX 7: AbortController for NGO fetch — Overpass API calls can take
    // 8-12 seconds on a good connection. On mobile they often just hang.
    // We give it 30 seconds then fall back with a clear message.
    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch('/ngo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      let data;
      try {
        data = await readApiJson(res, 'Could not read NGO result.');
      } catch (parseErr) {
        ngoResults.innerHTML = `<div class="ngo-error">${parseErr.message || 'Server returned an invalid response. Please try again.'}</div>`;
        return;
      }

      if (!res.ok) {
        ngoResults.innerHTML = `<div class="ngo-error">${data.error || 'Could not load NGO data. Please try again.'}</div>`;
        return;
      }

      renderNGOResults(data);
    } catch (err) {
      clearTimeout(timeoutId);
      const msg = err.name === 'AbortError'
        ? 'Request timed out. Try entering your city name instead.'
        : 'Could not load NGO data. Check your connection and try again.';
      ngoResults.innerHTML = `<div class="ngo-error">${msg}</div>`;
    }
  }

  // ── Render NGO results ──
  function renderNGOResults(data) {
    if (!ngoResults) return;

    if (!data.found && (!data.contacts || data.contacts.length === 0)) {
      ngoResults.innerHTML = `
        <div class="ngo-no-results">
          No contacts found near <strong>${data.city || 'your location'}</strong>.
          ${renderNationalCard(data.national)}
        </div>`;
      return;
    }

    const distTag = data.distance_km != null
      ? `<span class="ngo-distance-tag">${data.distance_km} km away</span>`
      : '';

    const mapsUrl = `https://www.google.com/maps/search/animal+hospital+vet+near+${encodeURIComponent(data.city)}`;

    let html = `
      <div class="ngo-results-header">
        <span class="ngo-city-name">${data.city}</span>
        ${distTag}
        <a class="ngo-maps-link" href="${mapsUrl}" target="_blank" rel="noopener">
          View on Google Maps &rarr;
        </a>
      </div>
      <div class="ngo-cards-grid">
        ${data.contacts.map(c => renderNGOCard(c)).join('')}
      </div>
      ${renderNationalCard(data.national)}`;

    ngoResults.innerHTML = html;
  }

  function renderNGOCard(c) {
    const typeTag = `<span class="ngo-tag ${c.type === 'vet' ? 'tag-vet' : 'tag-ngo'}">${c.type.toUpperCase()}</span>`;
    const emergencyTag = c.emergency ? `<span class="ngo-tag tag-emergency">Emergency</span>` : '';
    const address = c.address ? `<p class="ngo-address">${c.address}${c.maps_url ? ` — <a class="ngo-maps-inline" href="${c.maps_url}" target="_blank">Map</a>` : ''}</p>` : '';
    const phone   = c.phone   ? `<a class="contact-phone" href="tel:${c.phone}">${c.phone}</a>` : '';
    const email   = c.email   ? `<a class="contact-email" href="mailto:${c.email}">${c.email}</a>` : '';
    const website = c.website ? `<a class="contact-website" href="${c.website}" target="_blank" rel="noopener">${c.website.replace(/^https?:\/\//, '')}</a>` : '';

    return `
      <div class="ngo-card ${c.emergency ? 'ngo-card-emergency' : ''}">
        <div class="ngo-card-header">
          <span class="ngo-card-name">${c.name}</span>
          <span class="ngo-tags">${typeTag}${emergencyTag}</span>
        </div>
        ${address}
        <div class="ngo-contacts">${phone}${email}${website}</div>
      </div>`;
  }

  function renderNationalCard(nat) {
    if (!nat) return '';
    return `
      <div class="ngo-national-card">
        <span class="ngo-national-label">National Helpline</span>
        <span class="ngo-national-name">${nat.name}</span>
        <a class="ngo-national-phone" href="tel:${nat.phone}">${nat.phone}</a>
      </div>`;
  }

}); // ← closes DOMContentLoaded


// ── Report Card renderer (called from fetch handler above) ──
function showReportCard(data) {
  const reportSection = document.getElementById('report-card');
  const reportContent = document.getElementById('report-content');
  if (!reportSection || !reportContent) return;

  reportSection.style.display = 'block';

  // ── HEALTHY ──────────────────────────────────────────────────────────────────
  if (data.label === 'healthy') {
    reportContent.innerHTML = `
      <div class="report-top">

  <img
    src="${window.lastUploadedDogImage}"
    alt="Uploaded Dog"
    class="report-dog-img"
  >

  <div class="report-summary">
          <span class="status-badge status-healthy">Healthy</span>
          <p class="report-confidence">Confidence: <strong>${data.confidence}%</strong></p>
          <p class="report-note">
            The model found no visible signs of infection or disease.
            This is not a clinical diagnosis — if you have concerns, consult a vet.
          </p>
        </div>
      </div>

      <div class="report-disease-block" style="background:#f0fdf4; border-color:#bbf7d0;">
        <p class="report-section-title">What this means</p>
        <p style="font-size:0.92rem; color:#166534; line-height:1.7;">
          The dog appears to have a healthy coat, clear eyes, and no obvious
          skin lesions or injuries detected in this image. Continue regular
          check-ups, a balanced diet, and preventive parasite treatment.
        </p>
      </div>

      <div class="report-actions">
        <button class="btn-scan-again" onclick="window.openDogScanOptions && window.openDogScanOptions()">
          Scan Another Dog
        </button>
        <button class="btn-find-ngo" onclick="document.getElementById('ngo-section') && document.getElementById('ngo-section').scrollIntoView({behavior:'smooth'})">
          Find Nearby Vet
        </button>
        <button class="btn-back-top" onclick="window.scrollTo({top:0,behavior:'smooth'})">
          Back to Top
        </button>
      </div>`;

  // ── INFECTED ─────────────────────────────────────────────────────────────────
  } else {
    const d         = data.disease;
    const urgClass  = d.urgency === 'critical' ? 'urgency-high'
                    : d.urgency === 'high'     ? 'urgency-high'
                    : d.urgency === 'medium'   ? 'urgency-medium'
                    : 'urgency-low';

    const contagious = d.is_contagious
      ? '<span style="color:#c0392b; font-weight:600;">Yes — keep away from other animals</span>'
      : '<span style="color:#27ae60;">No</span>';
    const zoonotic = d.zoonotic
      ? '<span style="color:#c0392b; font-weight:600;">Yes — can spread to humans</span>'
      : '<span style="color:#27ae60;">No</span>';

    const top3HTML = d.top3_diseases.map(t => `
      <div class="meta-item">
        <span class="meta-label">${t.name}</span>
        <span class="meta-value">${t.pct}%</span>
      </div>`).join('');

    const symptomsHTML   = d.symptoms.map(s => `<p>${s}</p>`).join('');
    const firstAidHTML   = d.home_first_aid.map(s => `<p>${s}</p>`).join('');
    const treatmentHTML  = d.basic_cure.map(s => `<p>${s}</p>`).join('');

    reportContent.innerHTML = `
      <div class="report-top">

  <img
    src="${window.lastUploadedDogImage}"
    alt="Uploaded Dog"
    class="report-dog-img"
  >

  <div class="report-summary">
          <span class="status-badge status-infected">Infected</span>
          <p class="report-confidence">Confidence: <strong>${data.confidence}%</strong></p>
          <p class="report-note">AI analysis only — not a substitute for veterinary diagnosis.</p>
        </div>
      </div>

      <div class="report-disease-block">
        <p class="report-section-title">Detected Condition</p>
        <span class="disease-chip">${d.display_name}</span>

        <div class="report-meta-grid">
          <div class="meta-item">
            <span class="meta-label">Urgency</span>
            <span class="meta-value ${urgClass}">${d.urgency_label}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Contagious</span>
            <span class="meta-value">${contagious}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Zoonotic</span>
            <span class="meta-value">${zoonotic}</span>
          </div>
        </div>

        <p class="report-section-title">Top Differential Diagnoses</p>
        <div class="report-meta-grid">${top3HTML}</div>
      </div>

      <div class="report-disease-block" style="background:#fff8f6; border-color:#fad5ce;">
        <p class="report-section-title">Symptoms to Watch</p>
        <div class="report-symptoms">${symptomsHTML}</div>
      </div>

      <div class="report-disease-block" style="background:#fff8f6; border-color:#fad5ce;">
        <p class="report-section-title">Immediate First Aid</p>
        <div class="report-firstaid">${firstAidHTML}</div>
      </div>

      <div class="report-disease-block" style="background:#fff8f6; border-color:#fad5ce;">
        <p class="report-section-title">Recommended Treatment</p>
        <div class="report-firstaid">${treatmentHTML}</div>
      </div>

      <div class="report-actions">
        <button class="btn-scan-again" onclick="window.openDogScanOptions && window.openDogScanOptions()">
          Scan Another Dog
        </button>
        <button class="btn-find-ngo" onclick="
          const s = document.getElementById('ngo-section');
          if (s) s.scrollIntoView({ behavior: 'smooth' });
        ">
          Find Nearby Vet / NGO
        </button>
        <button class="btn-back-top" onclick="window.scrollTo({top:0,behavior:'smooth'})">
          Back to Top
        </button>
      </div>`;
  }

  reportSection.scrollIntoView({ behavior: 'smooth' });
}
