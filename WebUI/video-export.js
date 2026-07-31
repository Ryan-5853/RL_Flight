(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const modal = $('#rolloutExportModal');
  const openButton = $('#exportVideoButton');
  const startButton = $('#startRolloutExport');
  const cancelButton = $('#cancelRolloutExport');
  const closeButton = $('#closeRolloutExport');
  const progressPanel = $('#rolloutProgress');
  const stageLabel = $('#rolloutStage');
  const percentLabel = $('#rolloutPercent');
  const detailLabel = $('#rolloutProgressDetail');
  const progressBar = $('#rolloutProgressBar');
  const statusElement = $('#rolloutStatus');
  const sceneCanvas = $('#sceneCanvas');
  const ROLLOUT_API = '/api/runtime/rollouts';
  let activeJobId = null;
  let busy = false;
  let cancelRequested = false;

  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const nextPaint = () => new Promise(resolve => requestAnimationFrame(resolve));

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...(options.headers || {})
      }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const serverMessage = data.error || '';
      if (
        path.startsWith(ROLLOUT_API)
        && (
          response.status === 404
          || /unknown api endpoint/i.test(serverMessage)
        )
      ) {
        throw new Error(
          '当前后端进程尚未加载视频导出 API。请重启 python server.py，并确认反向代理允许 /api/runtime/rollouts。'
        );
      }
      throw new Error(serverMessage || `HTTP ${response.status}`);
    }
    return data;
  }

  function setStatus(message, kind = '') {
    statusElement.classList.toggle('error', kind === 'error');
    statusElement.classList.toggle('success', kind === 'success');
    statusElement.querySelector('span').textContent = message;
  }

  function setProgress(stage, percent, detail) {
    progressPanel.hidden = false;
    const bounded = Math.max(0, Math.min(100, percent));
    stageLabel.textContent = stage;
    percentLabel.textContent = `${Math.round(bounded)}%`;
    progressBar.style.width = `${bounded}%`;
    detailLabel.textContent = detail;
  }

  function setBusy(value) {
    busy = value;
    startButton.disabled = value;
    $('#rolloutFps').disabled = value;
    $('#rolloutResolution').disabled = value;
    $('#rolloutSaveData').disabled = value;
    closeButton.disabled = value;
    cancelButton.textContent = value ? '终止导出' : '取消';
  }

  function configuredDuration() {
    const input = document.querySelector(
      '[data-config="test"][data-path="task.episode_duration_s"]'
    );
    const duration = Number(input?.value);
    return Number.isFinite(duration) && duration > 0 ? duration : 30;
  }

  function updateDuration() {
    $('#rolloutDuration').textContent = `${configuredDuration().toFixed(1)} s`;
  }

  function openModal() {
    if (busy) return;
    updateDuration();
    progressPanel.hidden = true;
    setStatus('修改后的参数会在开始导出时自动校验；本地编码阶段请保持此页面在前台。');
    modal.hidden = false;
  }

  function closeModal() {
    if (!busy) modal.hidden = true;
  }

  async function cancelExport() {
    if (!busy) {
      closeModal();
      return;
    }
    cancelRequested = true;
    setStatus('正在终止离线采样或本地视频合成…');
    if (activeJobId) {
      await fetch(`${ROLLOUT_API}/${activeJobId}`, { method: 'DELETE' }).catch(() => {});
    }
  }

  function pauseInteractiveSession() {
    const runButton = $('#runButton');
    if (runButton?.classList.contains('running')) runButton.click();
  }

  async function withMaximumSceneResolution(callback) {
    const shell = $('.app-shell');
    if (!shell) return callback();
    const leftWasCollapsed = shell.classList.contains('left-collapsed');
    const rightWasCollapsed = shell.classList.contains('right-collapsed');
    shell.classList.add('left-collapsed', 'right-collapsed');
    // The workspace grid animates for 200 ms. Resizing before that transition
    // finishes leaves the canvas backing store at the old sidebar-constrained
    // aspect ratio, which the browser then stretches to the new CSS box.
    await wait(260);
    window.dispatchEvent(new Event('resize'));
    await nextPaint();
    try {
      return await callback();
    } finally {
      shell.classList.toggle('left-collapsed', leftWasCollapsed);
      shell.classList.toggle('right-collapsed', rightWasCollapsed);
      await wait(260);
      window.dispatchEvent(new Event('resize'));
      await nextPaint();
    }
  }

  async function pollRollout(jobId) {
    while (true) {
      if (cancelRequested) throw new Error('视频导出已取消');
      const data = await request(`${ROLLOUT_API}/${jobId}`);
      const job = data.job;
      const samplingProgress = Number(job.progress) || 0;
      setProgress(
        job.state === 'queued' ? '等待服务器采样资源…' : '服务器离线闭环采样',
        samplingProgress * 70,
        job.total_steps
          ? `${job.completed_steps.toLocaleString()} / ${job.total_steps.toLocaleString()} 个 500 Hz 控制周期`
          : '正在创建 SimEnv、控制器与虚拟飞手'
      );
      if (job.state === 'completed') return;
      if (job.state === 'failed') throw new Error(job.error || '离线 rollout 失败');
      if (job.state === 'cancelled') throw new Error('视频导出已取消');
      await wait(500);
    }
  }

  function downloadBlob(blob, filename) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  function firstBatch(value, width) {
    if (!Array.isArray(value)) return null;
    if (value.length === width && value.every(Number.isFinite)) return value;
    if (
      value.length
      && Array.isArray(value[0])
      && value[0].length === width
      && value[0].every(Number.isFinite)
    ) return value[0];
    return null;
  }

  function quaternionToEuler(quaternion) {
    const q = quaternion || [1, 0, 0, 0];
    const norm = Math.hypot(...q) || 1;
    const [w, x, y, z] = q.map(value => value / norm);
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [roll, pitch, yaw];
  }

  function wrapAngle(value) {
    return Math.atan2(Math.sin(value), Math.cos(value));
  }

  function buildAttitudeSeries(frames) {
    return frames.map(frame => {
      const actualQ = firstBatch(frame.truth?.attitude_q_wb, 4) || [1, 0, 0, 0];
      const referenceQ = firstBatch(frame.reference?.target_attitude_q_wb, 4)
        || frame.pilot?.target_attitude_q_wb
        || [1, 0, 0, 0];
      const actual = quaternionToEuler(actualQ);
      const target = quaternionToEuler(referenceQ);
      return {
        time: Number(frame.time_s) || 0,
        actual: actual.map(value => value * 180 / Math.PI),
        target: target.map(value => value * 180 / Math.PI),
        error: actual.map((value, axis) => wrapAngle(target[axis] - value) * 180 / Math.PI)
      };
    });
  }

  function roundedRect(ctx, x, y, width, height, radius = 6) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function drawLabel(ctx, text, x, y, color = '#718087', size = 8, align = 'left') {
    ctx.fillStyle = color;
    ctx.font = `600 ${size}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
  }

  function drawHeader(ctx, frame, rollout, index) {
    ctx.fillStyle = '#090d0f';
    ctx.fillRect(0, 0, 1280, 58);
    ctx.strokeStyle = '#253037';
    ctx.beginPath();
    ctx.moveTo(0, 57.5);
    ctx.lineTo(1280, 57.5);
    ctx.stroke();
    ctx.fillStyle = '#55d8e6';
    ctx.fillRect(24, 18, 22, 22);
    ctx.fillStyle = '#081012';
    ctx.fillRect(30, 24, 10, 10);
    drawLabel(ctx, 'RL FLIGHT', 58, 27, '#e3e9eb', 12);
    drawLabel(ctx, 'VIRTUAL PILOT · OFFLINE ROLLOUT', 58, 42, '#66747b', 7);
    const controllerType = frame.controller?.type || rollout.controller?.type || 'controller';
    drawLabel(ctx, controllerType.toUpperCase(), 1040, 25, '#b8dc74', 9, 'right');
    drawLabel(ctx, `${Number(frame.time_s).toFixed(2)} / ${Number(rollout.duration_s).toFixed(2)} s`, 1255, 42, '#9ba7ac', 9, 'right');
    const width = 215;
    ctx.fillStyle = '#202a2f';
    ctx.fillRect(1040, 32, width, 2);
    ctx.fillStyle = '#55d8e6';
    ctx.fillRect(1040, 32, width * (index / Math.max(1, rollout.frames.length - 1)), 2);
  }

  function drawScene(ctx, spatialStatus) {
    const x = 20, y = 76, width = 824, height = 470;
    roundedRect(ctx, x, y, width, height, 5);
    ctx.save();
    ctx.clip();
    ctx.fillStyle = '#080c0e';
    ctx.fillRect(x, y, width, height);
    if (sceneCanvas.width && sceneCanvas.height) {
      // Fill the video viewport without ever changing the source aspect
      // ratio. Crop only the excess edge around the centered aircraft.
      const sourceAspect = sceneCanvas.width / sceneCanvas.height;
      const targetAspect = width / height;
      let sourceX = 0;
      let sourceY = 0;
      let sourceWidth = sceneCanvas.width;
      let sourceHeight = sceneCanvas.height;
      if (sourceAspect > targetAspect) {
        sourceWidth = sceneCanvas.height * targetAspect;
        sourceX = (sceneCanvas.width - sourceWidth) / 2;
      } else if (sourceAspect < targetAspect) {
        sourceHeight = sceneCanvas.width / targetAspect;
        sourceY = (sceneCanvas.height - sourceHeight) / 2;
      }
      ctx.drawImage(
        sceneCanvas,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        x,
        y,
        width,
        height
      );
    }
    const gradient = ctx.createLinearGradient(x, y, x, y + height);
    gradient.addColorStop(0, 'rgba(5,9,11,.25)');
    gradient.addColorStop(.65, 'rgba(5,9,11,0)');
    gradient.addColorStop(1, 'rgba(5,9,11,.48)');
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, width, height);
    ctx.restore();
    ctx.strokeStyle = '#28353b';
    ctx.strokeRect(x + .5, y + .5, width - 1, height - 1);
    drawLabel(ctx, 'AIRCRAFT POSITION + ATTITUDE · CAM 01', x + 14, y + 20, '#55d8e6', 8);
    if (spatialStatus) {
      drawLabel(
        ctx,
        `AUTO SCALE  ${Number(spatialStatus.view_span_m).toFixed(1)} m VIEW  ·  GRID ${Number(spatialStatus.grid_step_m).toFixed(2)} m  ·  Δ ${Number(spatialStatus.displacement_m).toFixed(3)} m`,
        x + 14,
        y + height - 14,
        '#8fb5bb',
        7
      );
    }
  }

  function drawAttitudeCharts(ctx, series, index) {
    const colors = ['#55d8e6', '#b8dc74', '#eb9d61'];
    const names = ['ROLL ERROR', 'PITCH ERROR', 'YAW ERROR'];
    const panelX = 864, panelY = 76, panelW = 396, panelH = 470;
    ctx.fillStyle = '#0d1316';
    ctx.fillRect(panelX, panelY, panelW, panelH);
    ctx.strokeStyle = '#28353b';
    ctx.strokeRect(panelX + .5, panelY + .5, panelW - 1, panelH - 1);
    drawLabel(ctx, 'ATTITUDE TRACKING ERROR', panelX + 14, panelY + 20, '#aab5ba', 8);
    const chartX = panelX + 45;
    const chartW = panelW - 63;
    const chartH = 116;
    const startY = panelY + 40;
    const endTime = Math.max(series[series.length - 1]?.time || 1, 1e-3);
    for (let axis = 0; axis < 3; axis += 1) {
      const y = startY + axis * 137;
      const visible = series.slice(0, index + 1).map(item => item.error[axis]);
      const maxAbs = Math.max(5, ...visible.map(value => Math.abs(value)));
      const limit = Math.ceil(maxAbs / 5) * 5;
      ctx.fillStyle = '#0a0f12';
      ctx.fillRect(chartX, y, chartW, chartH);
      ctx.strokeStyle = '#1f292e';
      ctx.lineWidth = 1;
      for (let grid = 0; grid <= 4; grid += 1) {
        const gy = y + grid * chartH / 4;
        ctx.beginPath();
        ctx.moveTo(chartX, gy + .5);
        ctx.lineTo(chartX + chartW, gy + .5);
        ctx.stroke();
      }
      const zeroY = y + chartH / 2;
      ctx.strokeStyle = '#3a474d';
      ctx.beginPath();
      ctx.moveTo(chartX, zeroY);
      ctx.lineTo(chartX + chartW, zeroY);
      ctx.stroke();
      ctx.strokeStyle = colors[axis];
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      series.slice(0, index + 1).forEach((point, pointIndex) => {
        const px = chartX + point.time / endTime * chartW;
        const py = zeroY - Math.max(-limit, Math.min(limit, point.error[axis])) / limit * chartH * .45;
        if (pointIndex) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      });
      ctx.stroke();
      drawLabel(ctx, names[axis], panelX + 14, y + 12, colors[axis], 7);
      drawLabel(ctx, `+${limit}°`, chartX - 6, y + 7, '#536168', 6, 'right');
      drawLabel(ctx, `-${limit}°`, chartX - 6, y + chartH, '#536168', 6, 'right');
      const current = series[Math.min(index, series.length - 1)]?.error[axis] || 0;
      drawLabel(ctx, `${current >= 0 ? '+' : ''}${current.toFixed(2)}°`, chartX + chartW - 5, y + 13, '#d4dcde', 8, 'right');
    }
  }

  function drawStick(ctx, x, y, size, horizontal, vertical, labels, color) {
    ctx.fillStyle = '#090e11';
    ctx.fillRect(x, y, size, size);
    ctx.strokeStyle = '#2a363c';
    ctx.strokeRect(x + .5, y + .5, size - 1, size - 1);
    ctx.strokeStyle = '#202a2f';
    ctx.beginPath();
    ctx.moveTo(x + size / 2, y + 6);
    ctx.lineTo(x + size / 2, y + size - 6);
    ctx.moveTo(x + 6, y + size / 2);
    ctx.lineTo(x + size - 6, y + size / 2);
    ctx.stroke();
    const dotX = x + size / 2 + Math.max(-1, Math.min(1, horizontal)) * (size / 2 - 11);
    const dotY = y + size / 2 - Math.max(-1, Math.min(1, vertical)) * (size / 2 - 11);
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#071012';
    ctx.lineWidth = 2;
    ctx.stroke();
    drawLabel(ctx, labels[0], x + size / 2, y + size + 13, '#66747b', 6, 'center');
    drawLabel(ctx, labels[1], x - 7, y + size / 2 + 2, '#66747b', 6, 'right');
  }

  function drawPilotPanel(ctx, frame) {
    const x = 20, y = 562, width = 386, height = 138;
    ctx.fillStyle = '#0d1316';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = '#28353b';
    ctx.strokeRect(x + .5, y + .5, width - 1, height - 1);
    drawLabel(ctx, 'VIRTUAL PILOT STICKS', x + 13, y + 18, '#aab5ba', 8);
    const channels = frame.pilot?.channels || {};
    drawStick(ctx, x + 18, y + 31, 84, Number(channels.yaw) || 0, Number(channels.throttle) || -1, ['YAW', 'THR'], '#b8dc74');
    drawStick(ctx, x + 126, y + 31, 84, Number(channels.roll) || 0, -(Number(channels.pitch) || 0), ['ROLL', 'PITCH'], '#55d8e6');
    const values = [
      ['ROLL', channels.roll],
      ['PITCH', channels.pitch],
      ['YAW', channels.yaw],
      ['THROTTLE', channels.throttle]
    ];
    values.forEach(([name, value], row) => {
      const vy = y + 39 + row * 22;
      drawLabel(ctx, name, x + 235, vy, '#617077', 7);
      const number = Number(value) || 0;
      drawLabel(ctx, `${number >= 0 ? '+' : ''}${number.toFixed(3)}`, x + 365, vy, '#c6d0d3', 8, 'right');
      ctx.fillStyle = '#222c31';
      ctx.fillRect(x + 278, vy + 5, 87, 2);
      ctx.fillStyle = row === 3 ? '#b8dc74' : '#55d8e6';
      const center = x + 321.5;
      ctx.fillRect(number < 0 ? center + number * 43.5 : center, vy + 5, Math.abs(number) * 43.5, 2);
    });
  }

  function drawControllerPanel(ctx, frame) {
    const x = 422, y = 562, width = 422, height = 138;
    ctx.fillStyle = '#0d1316';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = '#28353b';
    ctx.strokeRect(x + .5, y + .5, width - 1, height - 1);
    drawLabel(ctx, 'CONTROLLER OUTPUT', x + 13, y + 18, '#aab5ba', 8);
    const command = firstBatch(frame.controller?.command, 5) || [0, 0, 0, 0, 0];
    const names = ['UPPER ROTOR', 'LOWER ROTOR', 'SERVO 1', 'SERVO 2', 'SERVO 3'];
    command.forEach((value, row) => {
      const cy = y + 36 + row * 19;
      drawLabel(ctx, names[row], x + 13, cy, '#68767c', 7);
      ctx.fillStyle = '#222c31';
      ctx.fillRect(x + 118, cy - 5, 228, 5);
      const normalized = row < 2
        ? Math.max(0, Math.min(1, Number(value)))
        : Math.max(-1, Math.min(1, Number(value)));
      if (row < 2) {
        ctx.fillStyle = row === 0 ? '#55d8e6' : '#b8dc74';
        ctx.fillRect(x + 118, cy - 5, normalized * 228, 5);
      } else {
        const center = x + 232;
        ctx.fillStyle = '#eb9d61';
        ctx.fillRect(normalized < 0 ? center + normalized * 114 : center, cy - 5, Math.abs(normalized) * 114, 5);
      }
      drawLabel(ctx, Number(value).toFixed(3), x + 404, cy, '#c4ced1', 7, 'right');
    });
  }

  function drawTelemetryFooter(ctx, frame, rollout, spatialStatus) {
    const position = firstBatch(frame.truth?.position_n, 3) || [0, 0, 0];
    const rates = firstBatch(frame.truth?.angular_velocity_b, 3) || [0, 0, 0];
    const height = -position[2];
    drawLabel(ctx, `POSITION NED  ${position.map(value => Number(value).toFixed(3)).join('  ')} m`, 865, 579, '#8b989d', 7);
    drawLabel(ctx, `HEIGHT  ${height.toFixed(3)} m`, 865, 600, '#c6d0d3', 9);
    drawLabel(ctx, `ANGULAR RATE  ${rates.map(value => Number(value).toFixed(3)).join('  ')} rad/s`, 865, 623, '#8b989d', 7);
    drawLabel(
      ctx,
      `DISPLACEMENT  ${Number(spatialStatus?.displacement_m || 0).toFixed(3)} m  ·  PILOT HEIGHT ERROR  ${Number(frame.pilot?.height_error_m || 0).toFixed(3)} m`,
      865,
      646,
      '#8b989d',
      7
    );
    drawLabel(ctx, `CONTROL ${rollout.control_hz} Hz  ·  VIDEO ${rollout.fps} FPS`, 865, 678, '#55d8e6', 7);
    if (rollout.termination && rollout.termination.reason !== 'episode_timeout') {
      drawLabel(ctx, `TERMINATED: ${String(rollout.termination.reason).toUpperCase()}`, 1257, 699, '#eb6d6d', 7, 'right');
    }
  }

  function drawComposite(canvas, frame, rollout, series, index) {
    const ctx = canvas.getContext('2d');
    const spatialStatus = window.RLFlightAircraft?.getSpatialStatus?.();
    const scale = canvas.width / 1280;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.fillStyle = '#070a0c';
    ctx.fillRect(0, 0, 1280, 720);
    drawHeader(ctx, frame, rollout, index);
    drawScene(ctx, spatialStatus);
    drawAttitudeCharts(ctx, series, index);
    drawPilotPanel(ctx, frame);
    drawControllerPanel(ctx, frame);
    drawTelemetryFooter(ctx, frame, rollout, spatialStatus);
  }

  function chooseRecorderMimeType() {
    const candidates = [
      'video/mp4;codecs=avc1.42E01E',
      'video/mp4',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm'
    ];
    return candidates.find(type => MediaRecorder.isTypeSupported(type)) || '';
  }

  async function encodeRolloutVideo(rollout, width, height) {
    if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) {
      throw new Error('当前浏览器不支持 Canvas 视频编码，请使用最新版 Chrome 或 Edge。');
    }
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    let stream = canvas.captureStream(0);
    let track = stream.getVideoTracks()[0];
    if (typeof track.requestFrame !== 'function') {
      track.stop();
      stream = canvas.captureStream(Number(rollout.fps || 30));
      track = stream.getVideoTracks()[0];
    }
    const mimeType = chooseRecorderMimeType();
    const recorder = new MediaRecorder(stream, {
      ...(mimeType ? { mimeType } : {}),
      videoBitsPerSecond: width >= 1920 ? 10_000_000 : 6_000_000
    });
    const chunks = [];
    recorder.addEventListener('dataavailable', event => {
      if (event.data.size) chunks.push(event.data);
    });
    const stopped = new Promise((resolve, reject) => {
      recorder.addEventListener('stop', resolve, { once: true });
      recorder.addEventListener('error', event => reject(event.error || new Error('视频编码失败')), { once: true });
    });
    const frames = rollout.frames || [];
    if (!frames.length) throw new Error('rollout 没有可导出帧');
    const series = buildAttitudeSeries(frames);
    const aircraftVisualization = window.RLFlightAircraft;
    const previousGridVisible = aircraftVisualization?.getSpatialStatus?.()?.grid_visible;
    aircraftVisualization?.setGridVisible?.(true);
    aircraftVisualization?.resetSpatialVisualization?.();
    const framePeriodMs = 1000 / Number(rollout.fps || 30);
    try {
      recorder.start(1000);
      const playbackStarted = performance.now();
      for (let index = 0; index < frames.length; index += 1) {
        if (cancelRequested) {
          recorder.stop();
          await stopped;
          throw new Error('视频导出已取消');
        }
        aircraftVisualization?.updateTelemetry(frames[index]);
        await nextPaint();
        drawComposite(canvas, frames[index], rollout, series, index);
        track.requestFrame?.();
        const percent = 70 + (index + 1) / frames.length * 30;
        setProgress(
          '浏览器本地合成视频',
          percent,
          `${index + 1} / ${frames.length} 帧 · ${width} × ${height} · 自适应位置网格`
        );
        const target = playbackStarted + (index + 1) * framePeriodMs;
        const remaining = target - performance.now();
        if (remaining > 1) await wait(remaining);
      }
      await wait(Math.max(40, framePeriodMs));
      recorder.stop();
      await stopped;
      track.stop();
      const finalType = recorder.mimeType || mimeType || 'video/webm';
      return {
        blob: new Blob(chunks, { type: finalType }),
        extension: finalType.includes('mp4') ? 'mp4' : 'webm'
      };
    } finally {
      if (recorder.state !== 'inactive') recorder.stop();
      if (track.readyState !== 'ended') track.stop();
      if (previousGridVisible !== undefined) {
        aircraftVisualization?.setGridVisible?.(previousGridVisible);
      }
    }
  }

  async function startExport() {
    if (busy) return;
    setBusy(true);
    cancelRequested = false;
    activeJobId = null;
    try {
      pauseInteractiveSession();
      await wait(150);
      setProgress('校验当前配置', 0, '正在生成 SimEnv 与控制器测试 YAML');
      setStatus('离线 rollout 期间可以保持远程连接空闲；控制闭环完全在服务器内部执行。');
      const capabilities = await request('/api/runtime/capabilities');
      if (capabilities.features?.offline_rollout !== true) {
        throw new Error(
          '当前运行的是旧版 WebUI 后端。请停止并重新启动 python server.py 后再导出；仅刷新浏览器不会重载 Python 路由。'
        );
      }
      const bundle = window.RLFlightConfig?.prepareSimulationStart();
      if (!bundle) throw new Error('无法生成当前仿真配置');
      const fps = Number($('#rolloutFps').value);
      const [width, height] = $('#rolloutResolution').value.split('x').map(Number);
      const checkpointPath = bundle.test.config.runtime?.checkpoint_path || null;
      const created = await request(ROLLOUT_API, {
        method: 'POST',
        body: JSON.stringify({
          simenv_yaml: bundle.simenv.yaml,
          test_yaml: bundle.test.yaml,
          checkpoint_path: checkpointPath,
          fps
        })
      });
      activeJobId = created.job.job_id;
      await pollRollout(activeJobId);
      setProgress('下载采样结果', 70, '闭环采样完成，正在一次性传输视频帧数据');
      const result = await request(`${ROLLOUT_API}/${activeJobId}/data`);
      const rollout = result.rollout;
      if ($('#rolloutSaveData').checked) {
        const dataBlob = new Blob(
          [JSON.stringify(rollout, null, 2)],
          { type: 'application/json;charset=utf-8' }
        );
        downloadBlob(dataBlob, `rlflight_rollout_${activeJobId.slice(0, 8)}.json`);
      }
      const encoded = await withMaximumSceneResolution(
        () => encodeRolloutVideo(rollout, width, height)
      );
      const controller = String(rollout.controller?.type || 'controller').replace(/[^a-z0-9_-]+/gi, '-');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      downloadBlob(encoded.blob, `rlflight_${controller}_${timestamp}.${encoded.extension}`);
      setProgress('导出完成', 100, `${encoded.extension.toUpperCase()} · ${(encoded.blob.size / 1024 / 1024).toFixed(1)} MiB`);
      setStatus('视频已生成并开始下载。采样闭环未经过浏览器或远程网络。', 'success');
    } catch (error) {
      const message = error?.message || String(error);
      setStatus(message, cancelRequested || message.includes('取消') ? '' : 'error');
      if (!cancelRequested) setProgress('导出失败', 0, message);
    } finally {
      activeJobId = null;
      setBusy(false);
      cancelRequested = false;
    }
  }

  openButton.addEventListener('click', openModal);
  startButton.addEventListener('click', startExport);
  cancelButton.addEventListener('click', cancelExport);
  closeButton.addEventListener('click', closeModal);
  modal.addEventListener('click', event => {
    if (event.target === modal) closeModal();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !modal.hidden) {
      if (busy) cancelExport();
      else closeModal();
    }
  });
})();
