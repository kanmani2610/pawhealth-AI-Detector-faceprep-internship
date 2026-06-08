// ── PawHealth AI — app.js ──

document.addEventListener('DOMContentLoaded', () => {

  // ── Smooth scroll for anchor links ──
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // ── "Scan a Dog" / "Analyse Now" / "Upload a Photo" → trigger file picker ──
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);

  const scanTriggers = document.querySelectorAll(
    '.nav-cta, .btn-primary, .btn-outline, .btn-white'
  );

  scanTriggers.forEach(btn => {
    // Only attach to buttons that are about scanning / uploading
    const label = btn.textContent.trim().toLowerCase();
    if (
      label.includes('scan') ||
      label.includes('analys') ||
      label.includes('upload') ||
      label.includes('start')
    ) {
      btn.addEventListener('click', () => fileInput.click());
    }
  });

  fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  try {
    showToast('Analysing dog image...');

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      showToast('Error: ' + data.error);
      return;
    }

    console.log(data);

    if (data.label === 'healthy') {
      showToast(
        `Healthy Dog (${data.confidence}% confidence)`
      );
    } else {
      showToast(
        `${data.disease.display_name} (${data.confidence}% confidence)`

      );

      showReportCard(data);
    }

  } catch (err) {
    console.error(err);
    showToast('Analysis failed');
  }

  fileInput.value = '';
});

  // ── "Contact Us" button ──
  document.querySelectorAll('.btn-pink').forEach(btn => {
    if (btn.textContent.trim().toLowerCase().includes('contact')) {
      btn.addEventListener('click', () => {
        showToast(' Contact form coming soon! Meanwhile call 1800-200-0167.');
      });
    }
  });

  // ── Sticky nav shadow on scroll ──
  const nav = document.querySelector('nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 10) {
      nav.style.boxShadow = '0 4px 30px rgba(190,24,93,0.13)';
    } else {
      nav.style.boxShadow = '0 2px 20px rgba(190,24,93,0.06)';
    }
  }, { passive: true });

  // ── Fade-in on scroll (IntersectionObserver) ──
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
      whiteSpace: 'nowrap',
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

  function showResultPopup(data) {
    const old = document.querySelector('.result-popup');
    if (old) old.remove();

    const popup = document.createElement('div');
    popup.className = 'result-popup';

    popup.innerHTML = `
      <div class="popup-inner">
        <button class="popup-close">&times;</button>
        <h2>${data.disease.display_name}</h2>
        <p>Confidence: ${data.confidence}%</p>
        <!-- Add more fields from data as needed -->
      </div>
    `;

    document.body.appendChild(popup);

    popup.querySelector('.popup-close')
      .addEventListener('click', () => popup.remove());

    popup.addEventListener('click', e => {
      if (e.target === popup) popup.remove();
    });
  }

}); // ← closes DOMContentLoaded
function showReportCard(data) {

  const reportSection = document.getElementById('report-card');
  const reportContent = document.getElementById('report-content');

  reportSection.style.display = 'block';

  reportContent.innerHTML = `
    <p><strong>Disease:</strong> ${data.disease.display_name}</p>

    <p><strong>Confidence:</strong> ${data.confidence}%</p>

    <p><strong>Urgency:</strong> ${data.disease.urgency_label}</p>

    <h3>Symptoms</h3>
    <ul>
      ${data.disease.symptoms.map(x => `<li>${x}</li>`).join('')}
    </ul>

    <h3>First Aid</h3>
    <ul>
      ${data.disease.home_first_aid.map(x => `<li>${x}</li>`).join('')}
    </ul>

    <h3>Treatment</h3>
    <ul>
      ${data.disease.basic_cure.map(x => `<li>${x}</li>`).join('')}
    </ul>
  `;

  reportSection.scrollIntoView({
    behavior: 'smooth'
  });
}