'use strict';

const $ = (selector, parent = document) => parent.querySelector(selector);
const formatNumber = (value, digits = 3) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
const percent = (value, digits = 2) => value === null || value === undefined ? '—' : `${formatNumber(Number(value) * 100, digits)}%`;

async function apiRequest(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { Accept: 'application/json', ...options.headers } });
  let body;
  try { body = await response.json(); } catch { throw new Error('Máy chủ trả về dữ liệu không hợp lệ. Vui lòng thử lại.'); }
  if (!response.ok) throw new Error(body.error?.message || body.message || `Không thể xử lý yêu cầu (HTTP ${response.status}).`);
  return body;
}

function setStatusBadge(ready, message) {
  document.querySelectorAll('[data-model-status]').forEach(badge => {
    badge.classList.remove('ready', 'unavailable');
    badge.classList.add(ready ? 'ready' : 'unavailable');
    badge.replaceChildren();
    const dot = document.createElement('span');
    dot.className = 'status-dot';
    badge.append(dot, document.createTextNode(message));
  });
}

function showModelAlert(message) {
  document.querySelectorAll('[data-model-alert]').forEach(alert => {
    alert.hidden = false;
    alert.replaceChildren(document.createTextNode(`${message} `));
    const code = document.createElement('code');
    code.textContent = 'python -m src.train';
    alert.append(document.createTextNode('Nếu chưa có artifacts, chạy '), code, document.createTextNode(' rồi khởi động lại ứng dụng.'));
  });
}

async function loadModel() {
  try {
    const data = await apiRequest('/api/model');
    setStatusBadge(data.model_loaded, data.model_loaded ? 'Mô hình đã sẵn sàng' : 'Mô hình chưa sẵn sàng');
    if (!data.model_loaded) showModelAlert(data.message || 'Chưa tìm thấy pipeline đã huấn luyện.');
    return data;
  } catch (error) {
    setStatusBadge(false, 'Không kết nối được máy chủ');
    showModelAlert(error.message);
    return null;
  }
}

function makePixels(container, values, illustration = false) {
  container.replaceChildren();
  const flat = values?.flat() || Array(64).fill(0);
  flat.forEach(value => {
    const pixel = document.createElement('span');
    const shade = Math.max(0, Math.min(1, Number(value) / 16));
    pixel.style.backgroundColor = illustration
      ? `rgb(${Math.round(31 + shade * 105)}, ${Math.round(53 + shade * 124)}, ${Math.round(87 + shade * 162)})`
      : `rgb(${Math.round(shade * 255)}, ${Math.round(shade * 255)}, ${Math.round(shade * 255)})`;
    container.append(pixel);
  });
}

function initializeHome() {
  const grid = $('#hero-digit');
  if (!grid) return;
  makePixels(grid, [
    [0, 0, 2, 11, 13, 4, 0, 0], [0, 1, 13, 7, 5, 14, 1, 0],
    [0, 3, 14, 4, 4, 14, 2, 0], [0, 0, 8, 15, 14, 7, 0, 0],
    [0, 2, 12, 8, 7, 13, 2, 0], [0, 5, 14, 0, 1, 13, 5, 0],
    [0, 2, 12, 5, 6, 14, 3, 0], [0, 0, 3, 12, 13, 4, 0, 0]
  ], true);
}

function initializeRecognition() {
  const canvas = $('#digit-canvas');
  if (!canvas) return;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const placeholder = $('#canvas-placeholder');
  const predictButton = $('#predict-button');
  const exampleButton = $('#example-button');
  const clearButton = $('#clear-button');
  const message = $('#prediction-status');
  let drawing = false;
  let inkPresent = false;
  let directSample = null;
  let examples = [];
  let exampleIndex = 0;
  let requestVersion = 0;
  let busy = false;
  let modelReady = false;
  let previousPoint;
  const bars = [];

  for (let digit = 0; digit < 10; digit += 1) {
    const row = document.createElement('div'); row.className = 'probability-row';
    const label = document.createElement('span'); label.className = 'probability-digit'; label.textContent = digit;
    const track = document.createElement('div'); track.className = 'probability-track'; track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div'); fill.className = 'probability-fill'; track.append(fill);
    const value = document.createElement('span'); value.className = 'probability-value'; value.textContent = '—';
    row.append(label, track, value); $('#probability-chart').append(row);
    bars.push({ row, fill, value });
  }

  function status(text, type = '') { message.textContent = text; message.className = `status-message ${type}`.trim(); }

  function resetResult() {
    requestVersion += 1;
    $('#prediction-number').textContent = '?'; $('#prediction-number').classList.remove('has-result');
    $('#result-caption').textContent = 'CHƯA CÓ DỰ ĐOÁN';
    $('#prediction-description').textContent = 'Chờ nét vẽ đầu tiên';
    $('#confidence-description').textContent = 'Kết quả và xác suất sẽ xuất hiện tại đây.';
    $('#probability-summary').textContent = 'Chưa có dữ liệu';
    $('#prediction-warning').hidden = true;
    bars.forEach(({ row, fill, value }) => { row.classList.remove('winner'); fill.style.width = '0%'; value.textContent = '—'; row.removeAttribute('aria-label'); });
  }

  function resetPreview() {
    makePixels($('#pixel-preview'), null);
    $('#pixel-preview').setAttribute('aria-label', 'Ảnh 8 × 8 sau tiền xử lý: chưa có dữ liệu');
    $('#preview-caption').textContent = 'Nhấn nhận dạng để xem ảnh sau tiền xử lý.';
  }

  function clear() {
    drawing = false; inkPresent = false; directSample = null;
    context.fillStyle = '#000'; context.fillRect(0, 0, 280, 280);
    placeholder.hidden = false;
    resetResult(); resetPreview();
    $('#preview-caption').textContent = 'Vẽ hoặc dùng mẫu để xem ảnh đầu vào.';
    status('Đã xoá nét vẽ. Vẽ một chữ số hoặc chọn mẫu Digits.');
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: Math.max(0, Math.min(280, (event.clientX - rect.left) * 280 / rect.width)), y: Math.max(0, Math.min(280, (event.clientY - rect.top) * 280 / rect.height)) };
  }

  function dot(at) {
    context.fillStyle = '#fff'; context.beginPath(); context.arc(at.x, at.y, 11, 0, Math.PI * 2); context.fill();
  }

  function stroke(from, to) {
    context.strokeStyle = '#fff'; context.lineWidth = 22; context.lineCap = 'round'; context.lineJoin = 'round';
    context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke();
  }

  canvas.addEventListener('pointerdown', event => {
    if (busy || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault(); canvas.setPointerCapture(event.pointerId);
    drawing = true; inkPresent = true; directSample = null; placeholder.hidden = true;
    resetResult(); resetPreview(); previousPoint = point(event); dot(previousPoint);
    status('Nét vẽ đã sẵn sàng. Nhấn “Nhận dạng chữ số” để xem kết quả.');
  });
  canvas.addEventListener('pointermove', event => {
    if (!drawing || busy) return;
    event.preventDefault();
    const events = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [event];
    (events.length ? events : [event]).forEach(item => { const next = point(item); stroke(previousPoint, next); previousPoint = next; });
  });
  function stopDrawing(event) {
    if (!drawing) return;
    drawing = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  }
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointercancel', stopDrawing);
  canvas.addEventListener('lostpointercapture', () => { drawing = false; });
  clearButton.addEventListener('click', clear);

  exampleButton.addEventListener('click', async () => {
    if (busy) return;
    exampleButton.disabled = true;
    try {
      if (!examples.length) {
        const data = await apiRequest('/api/examples'); examples = data.examples || [];
        if (!examples.length) throw new Error(data.message || 'Chưa có mẫu từ tập train. Hãy chạy pipeline tạo dữ liệu.');
      }
      const sample = examples[exampleIndex % examples.length]; exampleIndex += 1;
      clear(); directSample = sample; inkPresent = true; placeholder.hidden = true;
      for (let row = 0; row < 8; row += 1) for (let col = 0; col < 8; col += 1) {
        const shade = Math.round(sample.pixels[row][col] * 255 / 16);
        context.fillStyle = `rgb(${shade},${shade},${shade})`; context.fillRect(col * 35, row * 35, 35, 35);
      }
      makePixels($('#pixel-preview'), sample.pixels);
      $('#pixel-preview').setAttribute('aria-label', `Mẫu Digits từ tập train, nhãn thật ${sample.label}, ảnh 8 × 8.`);
      $('#preview-caption').textContent = `Mẫu train · nhãn thật ${sample.label} · gửi trực tiếp 64 pixel gốc.`;
      status(`Đã chọn mẫu Digits, nhãn thật ${sample.label}. Nhấn nhận dạng để kiểm tra mô hình.`);
    } catch (error) { status(error.message, 'error'); }
    finally { exampleButton.disabled = false; }
  });

  predictButton.addEventListener('click', async () => {
    if (busy) return;
    if (!inkPresent) { status('Khung vẽ đang trống. Vẽ một chữ số hoặc chọn mẫu Digits trước.', 'error'); canvas.focus(); return; }
    if (!modelReady) { status('Mô hình chưa sẵn sàng. Hãy kiểm tra thông báo phía trên và chạy pipeline huấn luyện.', 'error'); return; }
    const requestId = ++requestVersion;
    let payload;
    if (directSample) payload = { pixels: directSample.pixels };
    else {
      const image = context.getImageData(0, 0, 280, 280).data;
      payload = { canvas: Array.from({ length: 280 }, (_, row) => Array.from({ length: 280 }, (_, col) => image[(row * 280 + col) * 4])) };
    }
    busy = true; predictButton.disabled = true; exampleButton.disabled = true; clearButton.disabled = true;
    const label = $('span', predictButton); label.textContent = 'Đang phân loại…';
    predictButton.setAttribute('aria-busy', 'true'); status('Đang chuẩn hoá ảnh và chạy mạng MLP…');
    try {
      const result = await apiRequest('/api/digit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (requestId !== requestVersion) return;
      $('#prediction-number').textContent = result.prediction; $('#prediction-number').classList.add('has-result');
      $('#result-caption').textContent = 'CHỮ SỐ ĐƯỢC DỰ ĐOÁN';
      $('#prediction-description').textContent = `Có thể là chữ số ${result.prediction}`;
      $('#confidence-description').replaceChildren(document.createTextNode('Xác suất lớp dự đoán: '));
      const confidence = document.createElement('strong'); confidence.textContent = percent(result.confidence, 1); $('#confidence-description').append(confidence);
      $('#result-model').textContent = result.model_name || 'MLPClassifier';
      $('#probability-summary').textContent = 'Đầu ra của mô hình';
      result.probabilities.forEach(({ digit, probability }) => {
        if (!bars[digit]) return;
        const { row, fill, value } = bars[digit]; row.classList.toggle('winner', digit === result.prediction);
        fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`; value.textContent = percent(probability, 1);
        row.setAttribute('aria-label', `Chữ số ${digit}: ${percent(probability, 1)}`);
      });
      makePixels($('#pixel-preview'), result.pixels);
      $('#pixel-preview').setAttribute('aria-label', `Ảnh xám 8 × 8 đã gửi cho mô hình, giá trị pixel từ 0 đến 16; dự đoán ${result.prediction}.`);
      $('#preview-caption').textContent = directSample ? `Mẫu train · nhãn thật ${directSample.label} · pixel gốc được giữ nguyên.` : 'Ảnh canvas sau chuẩn hoá về 8 × 8, pixel 0–16.';
      const warnings = result.warnings || [];
      if (warnings.length) { $('#prediction-warning').textContent = warnings.join(' '); $('#prediction-warning').hidden = false; }
      status(`Dự đoán: ${result.prediction}, xác suất ${percent(result.confidence, 1)}.${directSample ? ` Nhãn thật của mẫu train: ${directSample.label}.` : ''}`, 'success');
    } catch (error) { if (requestId === requestVersion) status(error.message, 'error'); }
    finally { busy = false; predictButton.disabled = false; exampleButton.disabled = false; clearButton.disabled = false; label.textContent = 'Nhận dạng chữ số'; predictButton.removeAttribute('aria-busy'); }
  });
  clear(); status('Sẵn sàng. Vẽ một chữ số hoặc chọn mẫu từ tập train.');
  loadModel().then(data => { modelReady = Boolean(data?.model_loaded); if (data?.metadata?.model_name) $('#result-model').textContent = data.metadata.model_name; });
}

function tableCell(row, text, className = '') {
  const cell = document.createElement('td'); cell.textContent = text; if (className) cell.className = className; row.append(cell); return cell;
}

function addDetail(container, label, value, href) {
  const line = document.createElement('div'); const term = document.createElement('dt'); term.textContent = label;
  const definition = document.createElement('dd');
  if (href) { const link = document.createElement('a'); link.textContent = value; link.href = href; link.target = '_blank'; link.rel = 'noopener noreferrer'; definition.append(link); }
  else definition.textContent = value === undefined || value === null || value === '' ? 'Chưa có thông tin' : String(value);
  line.append(term, definition); container.append(line);
}

function showFigure(container, path, alt, fallback) {
  if (typeof path !== 'string' || !path.startsWith('/reports/figures/')) {
    if (fallback) fallback(); else { container.textContent = 'Chưa có biểu đồ trong artifacts.'; container.classList.add('chart-empty'); }
    return;
  }
  const image = document.createElement('img'); image.src = path; image.alt = alt; image.loading = 'lazy';
  image.addEventListener('error', () => { container.replaceChildren(); if (fallback) fallback(); else { container.textContent = 'Không tải được biểu đồ đã lưu.'; container.classList.add('chart-empty'); } }, { once: true });
  container.append(image);
}

function drawConfusionMatrix(container, matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 10) { container.textContent = 'Chưa có ma trận nhầm lẫn.'; container.classList.add('chart-empty'); return; }
  const max = Math.max(1, ...matrix.flat()); const table = document.createElement('table'); table.className = 'confusion-table';
  const caption = document.createElement('caption'); caption.textContent = 'Nhãn thật (hàng) / nhãn dự đoán (cột)'; table.append(caption);
  const head = document.createElement('thead'); const headRow = document.createElement('tr'); const corner = document.createElement('th'); corner.textContent = '↘'; headRow.append(corner);
  for (let digit = 0; digit < 10; digit += 1) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = digit; headRow.append(th); }
  head.append(headRow); table.append(head); const body = document.createElement('tbody');
  matrix.forEach((values, rowIndex) => { const row = document.createElement('tr'); const th = document.createElement('th'); th.scope = 'row'; th.textContent = rowIndex; row.append(th);
    values.forEach((value, colIndex) => { const cell = tableCell(row, value); cell.title = `Nhãn thật ${rowIndex}, dự đoán ${colIndex}: ${value} ảnh`; const intensity = value / max; cell.style.backgroundColor = `rgb(${Math.round(240 - intensity * 192)},${Math.round(245 - intensity * 149)},${Math.round(255 - intensity * 20)})`; cell.style.color = intensity > .5 ? 'white' : '#7790b1'; }); body.append(row);
  }); table.append(body); container.append(table);
}

async function initializeDashboard() {
  if (!$('#dashboard-content')) return;
  const data = await loadModel(); $('#dashboard-loading').hidden = true;
  if (!data?.evaluation?.models) {
    if (data?.model_loaded) showModelAlert('Pipeline đã sẵn sàng nhưng chưa có kết quả đánh giá. Chạy python -m src.evaluate để tạo báo cáo.');
    return;
  }
  const metadata = data.metadata || {};
  const evaluation = data.evaluation;
  const experiments = data.experiments || {};
  const models = evaluation.models; const mlp = models.mlp || {}; const baseline = models.baseline || {}; const figures = evaluation.figures || {};
  $('#dashboard-content').hidden = false;
  $('#metric-accuracy').textContent = percent(mlp.accuracy);
  $('#metric-f1').textContent = formatNumber(mlp.macro_f1, 4);
  $('#metric-logloss').textContent = formatNumber(mlp.log_loss, 4);
  $('#metric-test-count').textContent = formatNumber(evaluation.n_test ?? metadata?.split?.test, 0);
  if (metadata?.seed !== undefined) $('#metric-test-note').textContent = `Ảnh · stratified split · seed ${metadata.seed}`;
  [['baseline', baseline], ['mlp', mlp]].forEach(([kind, result]) => {
    const row = document.createElement('tr'); if (kind === 'mlp') row.className = 'selected';
    const nameCell = tableCell(row, ''); const wrap = document.createElement('span'); wrap.className = 'table-model-name'; wrap.textContent = result.name || (kind === 'mlp' ? 'MLPClassifier' : 'Logistic Regression');
    const tag = document.createElement('span'); tag.className = 'model-label'; tag.textContent = kind === 'mlp' ? 'Mô hình chính' : 'Baseline'; wrap.append(tag); nameCell.append(wrap);
    tableCell(row, percent(result.accuracy)); tableCell(row, formatNumber(result.macro_f1, 4)); tableCell(row, formatNumber(result.log_loss, 4)); $('#comparison-rows').append(row);
  });
  const delta = Number.isFinite(evaluation.accuracy_delta) ? evaluation.accuracy_delta : mlp.accuracy - baseline.accuracy;
  if (Number.isFinite(delta)) $('#comparison-conclusion').textContent = delta > 0 ? `MLP cao hơn baseline ${formatNumber(delta * 100, 2)} điểm phần trăm accuracy trên tập test này. Đây là kết quả của một split cố định; không suy rộng thành bảo đảm cho mọi nét viết.` : delta < 0 ? `MLP thấp hơn baseline ${formatNumber(Math.abs(delta) * 100, 2)} điểm phần trăm accuracy trên tập test này. Đối chiếu lỗi và dữ liệu trước khi kết luận.` : 'MLP và baseline có cùng accuracy trên tập test này. Macro-F1 và log loss cung cấp góc nhìn bổ sung.';
  showFigure($('#confusion-chart'), figures.confusion_matrix, 'Ma trận nhầm lẫn của Logistic Regression và MLP trên cùng tập test độc lập', () => drawConfusionMatrix($('#confusion-chart'), mlp.confusion_matrix));
  showFigure($('#learning-chart'), figures.learning_curve, 'Learning curve: trung bình và độ lệch chuẩn từ CV 3 fold chỉ trong tập train; không dùng tập validation bên ngoài hoặc tập test');
  const experimentList = Array.isArray(experiments) ? experiments : experiments.experiments || [];
  const selectedId = experiments.selected_experiment || metadata.selected_experiment;
  experimentList.forEach(experiment => {
    const config = experiment.configuration || {}; const metrics = experiment.validation || {}; const row = document.createElement('tr'); const selected = experiment.id === selectedId;
    if (selected) row.className = 'selected';
    const name = tableCell(row, experiment.name || experiment.id || 'Cấu hình');
    if (selected) { const label = document.createElement('span'); label.className = 'model-label'; label.style.marginLeft = '8px'; label.textContent = 'Đã chọn'; name.append(label); }
    if (experiment.kind === 'mlp' && experiment.eligible_for_selection === false) { const label = document.createElement('span'); label.className = 'model-label'; label.style.marginLeft = '8px'; label.textContent = 'Ablation'; name.append(label); }
    tableCell(row, Array.isArray(config.hidden_layer_sizes) ? config.hidden_layer_sizes.join(' → ') : '—');
    tableCell(row, config.alpha !== undefined ? formatNumber(config.alpha, 4) : '—');
    tableCell(row, typeof config.early_stopping === 'boolean' ? (config.early_stopping ? 'Bật' : 'Tắt') : '—');
    tableCell(row, percent(metrics.accuracy)); tableCell(row, formatNumber(metrics.macro_f1, 4)); tableCell(row, experiment.n_iter ?? '—'); $('#experiment-rows').append(row);
  });
  if (!experimentList.length) { const row = document.createElement('tr'); const cell = tableCell(row, 'Chưa có nhật ký thí nghiệm.'); cell.colSpan = 7; $('#experiment-rows').append(row); }
  $('#experiment-note').textContent = `Tiêu chí chọn: ${experiments.selection_metric || metadata.selection_metric || 'macro-F1 validation'}. Cấu hình early stopping tắt dùng phân tích ablation và không được chọn làm mô hình cuối.`;
  const pair = evaluation.most_confused_pair;
  $('#error-description').textContent = pair?.digits?.length === 2 ? `Cặp ${pair.digits[0]} ↔ ${pair.digits[1]} có ${pair.count} lượt nhầm giữa hai chữ số trên tập test. Các ảnh dưới đây giúp kiểm tra nét viết và dự đoán sai cụ thể.` : 'Quan sát các ảnh bị phân loại sai trên tập test để hiểu giới hạn của mô hình.';
  showFigure($('#error-chart'), figures.error_examples, 'Các ảnh trên tập test bị MLP phân loại sai, kèm nhãn thật và nhãn dự đoán');
  const config = metadata.configuration || {};
  const commonConfig = metadata.common_configuration || {};
  const modelDetails = $('#model-details');
  addDetail(modelDetails, 'Thuật toán', metadata.model_name || mlp.name || 'MLPClassifier');
  addDetail(modelDetails, 'Cấu hình đã chọn', selectedId);
  addDetail(modelDetails, 'Lớp ẩn', Array.isArray(config.hidden_layer_sizes) ? config.hidden_layer_sizes.join(' → ') : undefined);
  addDetail(modelDetails, 'Activation / solver', `${config.activation || commonConfig.activation || 'relu'} / ${config.solver || commonConfig.solver || 'adam'}`);
  addDetail(modelDetails, 'Alpha', config.alpha);
  addDetail(modelDetails, 'Early stopping', typeof config.early_stopping === 'boolean' ? (config.early_stopping ? 'Bật' : 'Tắt') : undefined);
  addDetail(modelDetails, 'Random seed', metadata.seed);
  addDetail(modelDetails, 'scikit-learn', metadata.sklearn_version);
  const trainingDate = metadata.trained_at ? new Date(metadata.trained_at) : null;
  addDetail(modelDetails, 'Huấn luyện lúc', trainingDate && !Number.isNaN(trainingDate.getTime()) ? `${trainingDate.toLocaleString('vi-VN', { timeZone: 'Asia/Bangkok' })} (UTC+7)` : undefined);
  const dataDetails = $('#data-details');
  addDetail(dataDetails, 'Nguồn dữ liệu', 'scikit-learn Digits ↗', 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html');
  addDetail(dataDetails, 'Nguồn gốc', 'UCI Optical Recognition of Handwritten Digits ↗', 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits');
  addDetail(dataDetails, 'Quy mô', '1.797 ảnh · 10 lớp · 64 feature');
  addDetail(dataDetails, 'Train / validation / test', metadata.split ? `${metadata.split.train} / ${metadata.split.validation} / ${metadata.split.test} ảnh` : undefined);
  addDetail(dataDetails, 'Đầu vào', 'Ảnh xám 8 × 8, pixel 0–16');
  addDetail(dataDetails, 'Tiền xử lý', typeof metadata.preprocessing === 'string' ? metadata.preprocessing : 'Pixel / 16 → [0, 1] trong Pipeline');
  addDetail(dataDetails, 'Bảo vệ test', 'Tách stratified trước huấn luyện; chọn tham số bằng validation');
  addDetail(dataDetails, 'SHA-256 dữ liệu', metadata.dataset_sha256);
  addDetail(dataDetails, 'SHA-256 mô hình', metadata.model_sha256);
  if (Array.isArray(metadata.limitations) && metadata.limitations.length) {
    $('#model-limitations').replaceChildren(); metadata.limitations.forEach(text => { const item = document.createElement('li'); item.textContent = text; $('#model-limitations').append(item); });
  }
}

document.addEventListener('DOMContentLoaded', () => { initializeHome(); initializeRecognition(); initializeDashboard(); });
