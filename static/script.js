// ===== DEEPFAKE DETECTION SYSTEM - JAVASCRIPT =====

const state = {
    currentFile: null,
    result: null,
    viewMode: 0
};

const fileInput     = document.getElementById('fileInput');
const fileNameDisp  = document.getElementById('fileName');
const analyzeBtn    = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        state.currentFile = e.target.files[0];
        fileNameDisp.textContent = state.currentFile.name;
    }
});

async function analyzeVideo() {
    const file = fileInput.files[0];
    if (!file) return alert("Select a video first!");

    analyzeBtn.disabled = true;
    document.getElementById('loader').style.display      = 'block';
    document.getElementById('statusText').style.display  = 'block';
    resultSection.style.display = 'none';

    document.getElementById('spatialBar').style.width  = '0%';
    document.getElementById('temporalBar').style.width = '0%';
    document.getElementById('artifactBar').style.width = '0%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) {
            alert("Analysis Error: " + data.error);
        } else {
            state.result = data;
            console.log("FULL API RESPONSE:", JSON.stringify(data));
            displayResults(data);
        }
    } catch (error) {
        alert("Error connecting to server. Check Flask terminal.");
    } finally {
        analyzeBtn.disabled = false;
        document.getElementById('loader').style.display     = 'none';
        document.getElementById('statusText').style.display = 'none';
    }
}

// Safely pull a metric from either data.metrics.key or flat data.key
function getMetric(data, key) {
    if (data.metrics && data.metrics[key] != null) return data.metrics[key];
    if (data[key] != null) return data[key];
    return 0;
}

function displayResults(data) {
    resultSection.style.display = 'block';

    const badge = document.getElementById('verdictBadge');
    badge.textContent = data.label;
    badge.className   = 'label-badge ' + (data.label === 'FAKE' ? 'badge-fake' : 'badge-real');

    const confCircle = document.getElementById('confidenceValue');
    confCircle.textContent       = data.confidence + '%';
    confCircle.style.borderColor = data.label === 'FAKE' ? 'var(--error-color)' : 'var(--success-color)';
    confCircle.style.boxShadow   = data.label === 'FAKE'
        ? '0 0 20px rgba(255,75,75,0.2), inset 0 0 10px rgba(255,75,75,0.1)'
        : '0 0 20px rgba(0,255,136,0.2), inset 0 0 10px rgba(0,255,136,0.1)';

    const s = getMetric(data, 'spatial');
    const t = getMetric(data, 'temporal');
    const a = getMetric(data, 'artifact');

    setTimeout(() => {
        document.getElementById('spatialVal').innerText    = s + '%';
        document.getElementById('spatialBar').style.width  = s + '%';
        document.getElementById('temporalVal').innerText   = t + '%';
        document.getElementById('temporalBar').style.width = t + '%';
        document.getElementById('artifactVal').innerText   = a + '%';
        document.getElementById('artifactBar').style.width = a + '%';
        document.getElementById('artifactBar').style.background =
            a > 50 ? 'var(--error-color)' : 'var(--success-color)';
    }, 300);

    const container    = document.getElementById('frameContainer');
    const toggleCamBtn = document.getElementById('toggleCamBtn');
    state.viewMode = 0;

    function renderFrames() {
        container.innerHTML = '';
        let framesToRender = data.frames;
        if (state.viewMode === 1 && data.cam_frames) framesToRender = data.cam_frames;
        if (state.viewMode === 2 && data.ela_frames) framesToRender = data.ela_frames;

        if (framesToRender) {
            framesToRender.forEach(f => {
                const img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + f;
                container.appendChild(img);
            });
        }
    }

    renderFrames();

    if (toggleCamBtn && data.cam_frames && data.ela_frames) {
        toggleCamBtn.style.display    = 'inline-block';
        toggleCamBtn.innerHTML        = '🔍 VIEW HEATMAPS';
        toggleCamBtn.style.background = 'transparent';

        toggleCamBtn.onclick = () => {
            state.viewMode = (state.viewMode + 1) % 3;
            if (state.viewMode === 0) {
                toggleCamBtn.innerHTML        = '🔍 VIEW HEATMAPS';
                toggleCamBtn.style.background = 'transparent';
            } else if (state.viewMode === 1) {
                toggleCamBtn.innerHTML        = '📡 VIEW ELA';
                toggleCamBtn.style.background = 'rgba(0,242,254,0.1)';
            } else {
                toggleCamBtn.innerHTML        = '👁️ VIEW ORIGINALS';
                toggleCamBtn.style.background = 'rgba(255,75,75,0.1)';
            }
            renderFrames();
        };
    } else if (toggleCamBtn) {
        toggleCamBtn.style.display = 'none';
    }

    setTimeout(() => {
        window.scrollTo({ top: resultSection.offsetTop - 30, behavior: 'smooth' });
    }, 100);
}

// ===== PDF GENERATOR =====
document.getElementById('downloadResultBtn').addEventListener('click', async function () {
    if (!state.result) return;

    const downloadBtn = this;
    downloadBtn.innerText = "⏳ GENERATING REPORT...";
    downloadBtn.disabled  = true;

    const element = document.getElementById('printReportTemplate');
    const mainUI  = document.querySelector('.container');

    try {
        const r    = state.result;
        const name = state.currentFile ? state.currentFile.name : "Target_Video.mp4";

        // 1. Populate text fields
        document.getElementById('pr_filename').innerText = name;
        document.getElementById('pr_date').innerText     = new Date().toLocaleString();
        document.getElementById('pr_case').innerText     = 'DF-' + Math.floor(10000 + Math.random() * 90000);

        const verdictEl            = document.getElementById('pr_verdict');
        verdictEl.innerText        = r.label;
        verdictEl.style.color      = r.label === 'FAKE' ? '#ff4b4b' : '#00cc6a';
        verdictEl.style.fontWeight = '800';

        document.getElementById('pr_confidence').innerText = r.confidence + '%';

        // 2. Metrics — handles data.metrics.x OR flat data.x
        document.getElementById('pr_spatial').innerText  = String(getMetric(r, 'spatial'));
        document.getElementById('pr_temporal').innerText = String(getMetric(r, 'temporal'));
        document.getElementById('pr_artifact').innerText = String(getMetric(r, 'artifact'));

        // 3. Inject evidence images
        const setImages = (prefix, frames) => {
            if (!frames) return;
            frames.slice(0, 5).forEach((f, i) => {
                const img = document.getElementById(`pr_${prefix}${i}`);
                if (img) {
                    img.src           = 'data:image/jpeg;base64,' + f;
                    img.style.display = 'block';
                }
            });
        };

        setImages('f', r.frames);
        setImages('c', r.cam_frames);
        setImages('e', r.ela_frames);

        // 4. Show template, hide main UI
        mainUI.style.display  = 'none';
        element.style.display = 'block';

        // 5. Wait for base64 images to fully decode & render
        await new Promise(resolve => setTimeout(resolve, 2000));

        // 6. Generate PDF
        const opt = {
            margin:   [10, 15, 15, 15],
            filename: `Forensic_Report_${name.split('.')[0]}.pdf`,
            image:    { type: 'jpeg', quality: 1.0 },
            html2canvas: {
                scale:   2,
                useCORS: true,
                logging: false,
                scrollY: 0,     // Prevents blank top space from scroll offset
                width:   750    // Match template width exactly
            },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        await html2pdf().set(opt).from(element).save();

    } catch (err) {
        console.error("PDF Error:", err);
        alert("Error generating report. Check the console.");
    } finally {
        // Always restore UI
        element.style.display = 'none';
        mainUI.style.display  = 'block';
        downloadBtn.innerHTML = "📥 DOWNLOAD FORENSIC REPORT";
        downloadBtn.disabled  = false;
    }
});

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ Deepfake Detection System loaded');
    checkModelStatus();
});

async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-status');
        const data     = await response.json();
        if (!data.model_loaded) {
            console.warn('⚠ Model not loaded — run train.py first');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}