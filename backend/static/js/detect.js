// detect.js - organized upload/preview + AJAX + last uploads

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const selectImage = document.getElementById('selectImage');
const selectVideo = document.getElementById('selectVideo');
const selectedTypeText = document.getElementById('selectedTypeText');

const previewPlaceholder = document.getElementById('previewPlaceholder');
const previewMedia = document.getElementById('previewMedia');
const previewBox = document.getElementById('previewBox');
const resultArea = document.getElementById('resultArea');
const resultBadge = document.getElementById('resultBadge');
const resultConfidence = document.getElementById('resultConfidence');
const loader = document.getElementById('loader');

const lastUploadsList = document.getElementById('lastUploadsList');

let selectedType = 'image';
let uploads = []; // in-memory last uploads (keeps recent items)

/* --- helper UI functions --- */
function setActiveCard(type){
  selectedType = type;
  selectedTypeText.textContent = (type === 'image') ? 'Photos' : 'Videos';
  selectImage.classList.toggle('active', type === 'image');
  selectVideo.classList.toggle('active', type === 'video');
  fileInput.accept = (type === 'image') ? 'image/*' : 'video/*';
  clearPreview();
  hideResult();
}

function clearPreview(){
  previewMedia.innerHTML = '';
  previewMedia.style.display = 'none';
  previewPlaceholder.style.display = 'block';
}

function showPreviewFile(file, type){
  previewPlaceholder.style.display = 'none';
  previewMedia.style.display = 'block';
  previewMedia.innerHTML = '';
  if(type === 'image'){
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    previewMedia.appendChild(img);
  } else {
    const vid = document.createElement('video');
    vid.controls = true;
    vid.src = URL.createObjectURL(file);
    vid.onloadeddata = () => URL.revokeObjectURL(vid.src);
    previewMedia.appendChild(vid);
  }
}

function showLoader(on){
  loader.style.display = on ? 'flex' : 'none';
  if(on) hideResult();
}

function showResult(label, conf){
  resultArea.style.display = 'block';
  resultBadge.textContent = label.toUpperCase();
  resultBadge.classList.remove('real','fake');
  if(label.toLowerCase() === 'real') resultBadge.classList.add('real');
  else if(label.toLowerCase() === 'fake') resultBadge.classList.add('fake');

  // confidence may be 0..1 or already percent
  let confText = '--';
  if(conf !== null && conf !== undefined){
    if(conf <= 1) confText = Math.round(conf * 100) + '%';
    else confText = conf + (String(conf).includes('%') ? '' : '%');
  }
  resultConfidence.textContent = confText;
  showLoader(false);
}

function hideResult(){
  resultArea.style.display = 'none';
}

/* --- last uploads handling --- */
function pushUploadRecord(record){
  uploads.unshift(record);
  if(uploads.length > 8) uploads.pop();
  renderLastUploads();
}

function renderLastUploads(){
  lastUploadsList.innerHTML = '';
  if(uploads.length === 0){
    const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'No uploads yet';
    lastUploadsList.appendChild(e); return;
  }
  uploads.forEach(r => {
    const row = document.createElement('div'); row.className = 'upload-row';
    const left = document.createElement('div');
    left.innerHTML = `<div class="name">${r.name}</div><div class="meta">${r.type} • ${r.time}</div>`;
    const right = document.createElement('div');
    right.innerHTML = `<div class="meta">${r.result} <span class="tag ${r.result.toLowerCase()}">●</span></div><div class="meta">${r.conf}</div>`;
    row.appendChild(left); row.appendChild(right);
    lastUploadsList.appendChild(row);
  });
}

/* --- upload / drag-drop events --- */
selectImage.addEventListener('click', () => setActiveCard('image'));
selectVideo.addEventListener('click', () => setActiveCard('video'));

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
dropzone.addEventListener('drop', (e) => {
  e.preventDefault(); dropzone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});

function handleFiles(fileList){
  if(!fileList || fileList.length === 0) return;
  const file = fileList[0];

  const trueType = file.type.startsWith('image') ? 'image' : (file.type.startsWith('video') ? 'video' : null);
  const useType = selectedType || trueType;
  if(useType === 'image' && !file.type.startsWith('image')) { alert('Please upload an image for Photos'); return; }
  if(useType === 'video' && !file.type.startsWith('video')) { alert('Please upload a video for Videos'); return; }

  // preview
  showPreviewFile(file, useType);

  // assemble request
  const url = (useType === 'image') ? '/predict_image' : '/predict_video';
  const fd = new FormData(); fd.append('file', file);

  showLoader(true);
  fetch(url, { method: 'POST', body: fd, headers: {'X-Requested-With': 'XMLHttpRequest'} })
    .then(r => {
      if(!r.ok) throw new Error('Server error');
      return r.json();
    })
    .then(data => {
      // expected JSON: { label: "real"/"fake", confidence: 0.89 } or {error: "..."}
      if(data.error){ alert('Error: ' + data.error); showLoader(false); return; }
      showResult(data.label || data.result || '—', (data.confidence !== undefined) ? data.confidence : data.conf || null);

      // push last uploads
      const rec = {
        name: file.name,
        type: useType,
        result: (data.label || data.result || '—'),
        conf: (data.confidence !== undefined) ? ((data.confidence <= 1) ? Math.round(data.confidence*100)+'%' : String(data.confidence)) : '--',
        time: new Date().toLocaleString()
      };
      pushUploadRecord(rec);
    })
    .catch(err => {
      showLoader(false);
      alert('Upload/analysis failed: ' + err.message);
    });
}

/* --- chat stub --- */
function toggleChat(){ alert('DeepGuard Assistant: Hi! Ask me about deepfakes, model usage, or demo results.'); }

/* --- init --- */
setActiveCard('image');
renderLastUploads();

// static/js/detect.js
(() => {
  // Utility: POST FormData and return JSON
  async function postFile(url, file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(url, {
      method: 'POST',
      body: fd,
      headers: { 'X-Requested-With': 'XMLHttpRequest' } // server expects this to return JSON
    });
    return res.json();
  }

  // After prediction, refresh visualization data and broadcast it
  async function refreshVizAndBroadcast() {
    try {
      const r = await fetch('/api/visualization-data');
      if (!r.ok) return;
      const payload = await r.json();
      // dispatch a custom event so visualization page (if open) updates instantly
      window.dispatchEvent(new CustomEvent('viz-update', { detail: payload }));
    } catch (e) {
      console.warn('Could not refresh visualization data:', e);
    }
  }

  // Bind upload form(s) if present (image & video)
  function init() {
    // Image form id="image-upload-form" and <input type="file" id="image-file">
    const imgForm = document.getElementById('image-upload-form');
    if (imgForm) {
      imgForm.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const input = document.getElementById('image-file');
        if (!input || !input.files.length) return alert('Select an image');
        const file = input.files[0];
        try {
          const result = await postFile('/predict_image', file);
          // show result on page (if you have an element)
          const out = document.getElementById('image-result');
          if (out) out.innerText = JSON.stringify(result, null, 2);
        } catch (err) {
          console.error(err);
        }
        await refreshVizAndBroadcast();
      });
    }

    // Video form id="video-upload-form" and <input type="file" id="video-file">
    const vidForm = document.getElementById('video-upload-form');
    if (vidForm) {
      vidForm.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const input = document.getElementById('video-file');
        if (!input || !input.files.length) return alert('Select a video');
        const file = input.files[0];
        try {
          const result = await postFile('/predict_video', file);
          const out = document.getElementById('video-result');
          if (out) out.innerText = JSON.stringify(result, null, 2);
        } catch (err) {
          console.error(err);
        }
        await refreshVizAndBroadcast();
      });
    }
  }
  

  // init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else init();

})();

const bc = new BroadcastChannel('truthlens_channel');
bc.postMessage({ type:'instant', payload: instant });