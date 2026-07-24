(() => {
  'use strict';

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const canvas = $('#sceneCanvas');
  const ctx = canvas.getContext('2d');

  const state = {
    running: false,
    elapsed: 0,
    episode: 42,
    lastFrame: performance.now(),
    yaw: 0.04,
    pitch: -0.08,
    roll: 0.02,
    cameraYaw: -0.72,
    cameraPitch: 0.5,
    zoom: 1,
    grid: true,
    topView: false,
    dragging: false,
    dragX: 0,
    dragY: 0
  };

  const MODEL_SCALE = 4;
  const aircraftVisual = {
    geometry: {
      centerOfMass: [0, 0, 0.08],
      directThrustCenter: [0, 0, 0.20],
      neutralThrustDirection: [0, 0, -1],
      gridCenters: [[0.10, 0, 0.25], [-0.05, 0.087, 0.25], [-0.05, -0.087, 0.25]],
      gridAxes: [[1, 0, 0], [-0.5, 0.8660254, 0], [-0.5, -0.8660254, 0]]
    },
    gridMoments: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    totalMoment: [0, 0, 0],
    motorSpeed: [0, 0],
    rotorPhase: [0, 0],
    servoAngle: [0, 0, 0],
    lastBackendAt: 0
  };

  function resizeCanvas() {
    const box = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  const rotateX = ([x, y, z], a) => [x, y * Math.cos(a) - z * Math.sin(a), y * Math.sin(a) + z * Math.cos(a)];
  const rotateY = ([x, y, z], a) => [x * Math.cos(a) + z * Math.sin(a), y, -x * Math.sin(a) + z * Math.cos(a)];
  const rotateZ = ([x, y, z], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a), z];

  function worldProject(point) {
    const box = canvas.getBoundingClientRect();
    let p = state.topView ? rotateX(point, Math.PI / 2) : rotateX(rotateZ(point, state.cameraYaw), state.cameraPitch);
    const scale = Math.min(box.width, box.height) * 0.105 * state.zoom;
    return [box.width / 2 + p[0] * scale, box.height * 0.54 - p[2] * scale - p[1] * scale * 0.08, p[1]];
  }

  function craftProject(point) {
    // SimEnv 使用 FRD/NED（+Z 向下），Canvas 使用常见的 +Z 向上视觉空间。
    // 所有机体几何与向量必须在姿态旋转前统一转换，避免电机/格栅上下颠倒。
    let p = [point[0], point[1], -point[2]];
    p = rotateX(p, state.roll);
    p = rotateY(p, state.pitch);
    p = rotateZ(p, state.yaw);
    return worldProject(p);
  }

  function line(a, b, color, width = 1, alpha = 1) {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawGrid() {
    if (!state.grid) return;
    for (let n = -5; n <= 5; n += 0.5) {
      const major = Number.isInteger(n);
      line(worldProject([n, -5, 0]), worldProject([n, 5, 0]), major ? '#263137' : '#171f23', major ? 0.8 : 0.5, major ? .8 : .65);
      line(worldProject([-5, n, 0]), worldProject([5, n, 0]), major ? '#263137' : '#171f23', major ? 0.8 : 0.5, major ? .8 : .65);
    }
  }

  function drawAxis() {
    const origin = worldProject([0, 0, 0]);
    const axes = [
      { p: [2.6, 0, 0], c: '#e66d6d', t: 'X' },
      { p: [0, 2.6, 0], c: '#b8dc74', t: 'Y' },
      { p: [0, 0, -2.6], c: '#6997f0', t: 'Z↓' }
    ];
    axes.forEach(({ p, c, t }) => {
      const end = worldProject(p);
      line(origin, end, c, 1.4, .9);
      ctx.fillStyle = c;
      ctx.font = '600 9px IBM Plex Mono';
      ctx.fillText(t, end[0] + 5, end[1] + 3);
    });
  }

  function polygon(points, fill, stroke, width = 1) {
    const projected = points.map(craftProject);
    ctx.beginPath();
    projected.forEach((p, index) => index ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function circleAt(z, radius = 1.2, segments = 36) {
    return Array.from({ length: segments }, (_, index) => {
      const angle = index / segments * Math.PI * 2;
      return [Math.cos(angle) * radius, Math.sin(angle) * radius, z];
    });
  }

  function polyline3d(points, color, width = 1, alpha = 1, closed = false) {
    const projected = points.map(craftProject);
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    projected.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
    if (closed) ctx.closePath();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.globalAlpha = 1;
  }

  function drawRotor(z, index) {
    const ring = circleAt(z, .86, 30);
    polyline3d(ring, index ? '#78955f' : '#3d8690', 1, .75, true);
    const phase = aircraftVisual.rotorPhase[index];
    for (let blade = 0; blade < 2; blade += 1) {
      const angle = phase + blade * Math.PI / 2;
      const direction = [Math.cos(angle) * .78, Math.sin(angle) * .78, z];
      line(craftProject([-direction[0], -direction[1], z]), craftProject(direction), index ? '#9fc16c' : '#55d8e6', 1.2, .72);
    }
    const hub = craftProject([0, 0, z]);
    ctx.beginPath(); ctx.arc(hub[0], hub[1], 3.2, 0, Math.PI * 2); ctx.fillStyle = index ? '#9fc16c' : '#55d8e6'; ctx.fill();
  }

  function drawGrids() {
    const colors = ['#55d8e6', '#b8dc74', '#eb9d61'];
    const centerOfMass = aircraftVisual.geometry.centerOfMass.map(value => value * MODEL_SCALE);
    aircraftVisual.geometry.gridCenters.forEach((rawCenter, index) => {
      const center = rawCenter.map(value => value * MODEL_SCALE);
      line(craftProject(centerOfMass), craftProject(center), colors[index], .7, .28);
      const rawAxis = aircraftVisual.geometry.gridAxes[index] || [Math.cos(index * Math.PI * 2 / 3), Math.sin(index * Math.PI * 2 / 3), 0];
      const axisMagnitude = Math.hypot(...rawAxis) || 1;
      const axis = rawAxis.map(value => value / axisMagnitude);
      const servo = aircraftVisual.servoAngle[index] || 0;
      const guide = rotateVectorAroundAxis(aircraftVisual.geometry.neutralThrustDirection, axis, servo);

      // 细线表示格栅旋转轴；栅片平面由旋转轴和导流方向张成。
      line(
        craftProject(center.map((value, component) => value - axis[component] * .34)),
        craftProject(center.map((value, component) => value + axis[component] * .34)),
        colors[index], 1, .42
      );
      for (let slat = -1; slat <= 1; slat += 1) {
        const offset = slat * .14;
        const half = .28;
        const slatCenter = center.map((value, component) => value + axis[component] * offset);
        const start = slatCenter.map((value, component) => value - guide[component] * half);
        const end = slatCenter.map((value, component) => value + guide[component] * half);
        line(craftProject(start), craftProject(end), colors[index], 1.5, .9);
      }
      const point = craftProject(center);
      ctx.beginPath(); ctx.arc(point[0], point[1], 2.5, 0, Math.PI * 2); ctx.fillStyle = colors[index]; ctx.fill();
    });
  }

  function rotateVectorAroundAxis(vector, axis, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    const dot = vector.reduce((sum, value, index) => sum + value * axis[index], 0);
    const cross = [
      axis[1] * vector[2] - axis[2] * vector[1],
      axis[2] * vector[0] - axis[0] * vector[2],
      axis[0] * vector[1] - axis[1] * vector[0]
    ];
    return vector.map((value, index) => value * cosine + cross[index] * sine + axis[index] * dot * (1 - cosine));
  }

  function magnitude(vector) { return Math.hypot(vector[0] || 0, vector[1] || 0, vector[2] || 0); }

  function drawMomentArrow(originRaw, vector, color, label, maxMagnitude, strong = false) {
    const mag = magnitude(vector);
    if (mag < 1e-7) return;
    const origin = originRaw.map(value => value * MODEL_SCALE);
    const relative = Math.min(1, mag / Math.max(maxMagnitude, 1e-9));
    const length = (strong ? .72 : .48) + relative * (strong ? .82 : .62);
    const direction = vector.map(value => value / mag);
    const end = origin.map((value, index) => value + direction[index] * length);
    const a = craftProject(origin), b = craftProject(end);
    line(a, b, color, strong ? 2.3 : 1.7, .95);
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const head = strong ? 9 : 7;
    ctx.beginPath(); ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - Math.cos(angle - .5) * head, b[1] - Math.sin(angle - .5) * head);
    ctx.lineTo(b[0] - Math.cos(angle + .5) * head, b[1] - Math.sin(angle + .5) * head);
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
    ctx.fillStyle = color; ctx.font = `${strong ? 600 : 500} 8px IBM Plex Mono`;
    ctx.fillText(`${label} ${mag.toFixed(3)}`, b[0] + 6, b[1] - 5);
  }

  function updateMomentFallback(now) {
    if (now - aircraftVisual.lastBackendAt < 600) return;
    if (!state.running) {
      aircraftVisual.gridMoments = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      aircraftVisual.totalMoment = [0, 0, 0];
      aircraftVisual.motorSpeed = [0, 0];
      return;
    }
    const t = now / 1000;
    aircraftVisual.gridMoments = [
      [0.018 * Math.sin(t * 1.3), 0.025 * Math.cos(t * .9), 0.004],
      [-0.020 * Math.sin(t * 1.1 + 1), 0.016 * Math.cos(t * 1.2), -0.003],
      [0.014 * Math.sin(t * .8 - .7), -0.022 * Math.cos(t), 0.002]
    ];
    aircraftVisual.totalMoment = aircraftVisual.gridMoments.reduce((sum, item) => sum.map((value, axis) => value + item[axis]), [0, 0, .006 * Math.sin(t)]);
    aircraftVisual.servoAngle = aircraftVisual.gridMoments.map(moment => Math.max(-.35, Math.min(.35, magnitude(moment) * 5)));
    aircraftVisual.motorSpeed = [1180 + Math.sin(t) * 60, 1120 + Math.cos(t * .8) * 55];
  }

  function drawMoments() {
    const moments = [...aircraftVisual.gridMoments, aircraftVisual.totalMoment];
    const maxMagnitude = Math.max(...moments.map(magnitude), 1e-6);
    const colors = ['#55d8e6', '#b8dc74', '#eb9d61'];
    aircraftVisual.gridMoments.forEach((moment, index) => drawMomentArrow(aircraftVisual.geometry.gridCenters[index], moment, colors[index], `M${index + 1}`, maxMagnitude));
    line(
      craftProject(aircraftVisual.geometry.centerOfMass.map(value => value * MODEL_SCALE)),
      craftProject(aircraftVisual.geometry.directThrustCenter.map(value => value * MODEL_SCALE)),
      '#e8edf0', .8, .25
    );
    drawMomentArrow(aircraftVisual.geometry.directThrustCenter, aircraftVisual.totalMoment, '#e8edf0', 'ΣM', maxMagnitude, true);
    aircraftVisual.gridMoments.forEach((moment, index) => setText(`#moment${index + 1}Value`, `${magnitude(moment).toFixed(3)} N·m`));
    setText('#momentTotalValue', `${magnitude(aircraftVisual.totalMoment).toFixed(3)} N·m`);
  }

  function drawCraft() {
    // 机体为直径:高度约 2:1 的短圆筒，z 轴沿 FRD 向下。
    const top = circleAt(0, 1.2), bottom = circleAt(1.2, 1.2);
    drawRotor(.30, 0);
    drawRotor(.58, 1);
    for (let index = 0; index < top.length; index += 1) {
      const next = (index + 1) % top.length;
      polygon([top[index], top[next], bottom[next], bottom[index]], index % 2 ? 'rgba(18,34,39,.25)' : 'rgba(25,45,51,.30)', '#29434a', .45);
    }
    polygon(top, 'rgba(18,31,36,.48)', '#4f7b83', 1.2);
    polyline3d(bottom, '#42636b', 1, .9, true);
    drawGrids();

    // 质心和机头标记用于观察配置中的几何偏置与姿态方向。
    const com = craftProject(aircraftVisual.geometry.centerOfMass.map(value => value * MODEL_SCALE));
    ctx.beginPath(); ctx.arc(com[0], com[1], 4, 0, Math.PI * 2); ctx.fillStyle = '#f2c86f'; ctx.fill();
    ctx.strokeStyle = '#17110a'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = '#b99b5c'; ctx.font = '600 7px IBM Plex Mono'; ctx.fillText('CG', com[0] + 6, com[1] - 4);
    const nose = craftProject([1.2, 0, .25]);
    ctx.beginPath(); ctx.arc(nose[0], nose[1], 3.5, 0, Math.PI * 2); ctx.fillStyle = '#eb6d6d'; ctx.fill();
    ctx.fillStyle = '#b96565'; ctx.fillText('FRONT', nose[0] + 6, nose[1] + 3);
    drawMoments();
  }

  function drawTarget() {
    const radius = 2.05;
    const center = worldProject([0, 0, .1]);
    ctx.beginPath();
    for (let i = 0; i <= 64; i++) {
      const p = worldProject([Math.cos(i / 64 * Math.PI * 2) * radius, Math.sin(i / 64 * Math.PI * 2) * radius, .1]);
      i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    }
    ctx.strokeStyle = '#2f484e'; ctx.setLineDash([4, 6]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(center[0], center[1], 3, 0, Math.PI * 2); ctx.fillStyle = '#55d8e6'; ctx.fill();
  }

  function drawScene() {
    const box = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, box.width, box.height);
    drawGrid();
    drawTarget();
    drawAxis();
    drawCraft();
  }

  function setText(id, text) { $(id).textContent = text; }
  function setOutput(name, value, motor = false) {
    const normalized = motor ? Math.max(0, Math.min(1, value)) : Math.max(-1, Math.min(1, value));
    setText(`#${name}Value`, normalized.toFixed(2));
    const bar = $(`#${name}Bar`);
    if (motor) {
      bar.style.width = `${normalized * 100}%`;
      bar.style.left = '0';
    } else {
      bar.style.width = `${Math.abs(normalized) * 50}%`;
      bar.style.left = normalized < 0 ? `${50 - Math.abs(normalized) * 50}%` : '50%';
    }
  }

  function updateTelemetry(now) {
    if (!state.running) return;
    const dt = Math.min((now - state.lastFrame) / 1000, .1);
    state.elapsed += dt;
    if (state.elapsed >= 30) { state.elapsed = 0; state.episode += 1; }

    const targetR = Number($('#targetRoll')?.value || 0) * Math.PI / 180;
    const targetP = Number($('#targetPitch')?.value || 0) * Math.PI / 180;
    const targetY = Number($('#targetYaw')?.value || 0) * Math.PI / 180;
    if (now - aircraftVisual.lastBackendAt >= 600) {
      state.roll += (targetR - state.roll) * dt * 1.8 + Math.sin(now / 700) * dt * .01;
      state.pitch += (targetP - state.pitch) * dt * 1.5 + Math.sin(now / 930) * dt * .008;
      state.yaw += (targetY - state.yaw) * dt * .8;
    }

    const r = state.roll * 180 / Math.PI, p = state.pitch * 180 / Math.PI, y = state.yaw * 180 / Math.PI;
    setText('#rollValue', `${r >= 0 ? '+' : ''}${r.toFixed(2)}°`);
    setText('#pitchValue', `${p >= 0 ? '+' : ''}${p.toFixed(2)}°`);
    setText('#yawValue', `${y >= 0 ? '+' : ''}${y.toFixed(2)}°`);
    const cr = Math.cos(state.roll / 2), sr = Math.sin(state.roll / 2), cp = Math.cos(state.pitch / 2), sp = Math.sin(state.pitch / 2), cy = Math.cos(state.yaw / 2), sy = Math.sin(state.yaw / 2);
    const q = [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy];
    setText('#quatValue', `[${q.map(v => v.toFixed(3)).join(', ')}]`);
    const motorL = .62 + Math.sin(now / 420) * .035 - state.roll * .12;
    const motorR = .62 + Math.sin(now / 450) * .035 + state.roll * .12;
    setOutput('motorL', motorL, true); setOutput('motorR', motorR, true);
    setOutput('servoX', (targetR - state.roll) * 1.4); setOutput('servoY', (targetP - state.pitch) * 1.4); setOutput('servoZ', (targetY - state.yaw) * .7);
    setText('#angularVelocity', `${(targetR-state.roll).toFixed(2)}\u00a0\u00a0 ${(targetP-state.pitch).toFixed(2)}\u00a0\u00a0 ${(targetY-state.yaw).toFixed(2)} rad/s`);
    if (now - aircraftVisual.lastBackendAt >= 600) {
      setText('#latencyValue', `${(0.31 + Math.sin(now / 800) * .04).toFixed(2)} ms`);
    }
    setText('#episodeNumber', String(state.episode).padStart(4, '0'));
    setText('#episodeTime', `00:${String(Math.floor(state.elapsed)).padStart(2, '0')}.${String(Math.floor(state.elapsed % 1 * 1000)).padStart(3, '0')} / 30s`);
    $('#episodeProgress').style.width = `${state.elapsed / 30 * 100}%`;
  }

  function frame(now) {
    updateTelemetry(now);
    updateMomentFallback(now);
    const visualDt = Math.min((now - state.lastFrame) / 1000, .05);
    aircraftVisual.rotorPhase[0] += aircraftVisual.motorSpeed[0] * visualDt * .035;
    aircraftVisual.rotorPhase[1] -= aircraftVisual.motorSpeed[1] * visualDt * .035;
    drawScene();
    state.lastFrame = now;
    requestAnimationFrame(frame);
  }

  function setRunning(running) {
    state.running = running;
    $('#runButton').classList.toggle('running', running);
    setText('#runLabel', running ? '暂停仿真' : '启动仿真');
    setText('#simStateLabel', running ? 'RUNNING' : 'PAUSED');
    setText('#stateSymbol', running ? '▶' : 'Ⅱ');
    setText('#stateTitle', running ? '仿真运行中' : '仿真已暂停');
    setText('#stateHint', running ? 'Policy + SimEnv · CPU' : '状态已保留，可继续运行');
  }

  function resetSimulation() {
    state.elapsed = 0; state.roll = .02; state.pitch = -.08; state.yaw = .04;
    $('#episodeProgress').style.width = '0';
    setText('#episodeTime', '00:00.000 / 30s');
    window.dispatchEvent(new CustomEvent('rlflightsimulationreset'));
  }

  $('#runButton').addEventListener('click', () => {
    let bundle = null;
    if (!state.running) {
      try {
        bundle = window.RLFlightConfig?.prepareSimulationStart();
      } catch (error) {
        const indicator = $('#dirtyIndicator');
        indicator.classList.add('dirty');
        indicator.querySelector('span').textContent = error.message;
        return;
      }
    }
    const nextRunning = !state.running;
    setRunning(nextRunning);
    window.dispatchEvent(new CustomEvent(nextRunning ? 'rlflightsimulationstart' : 'rlflightsimulationpause', {
      detail: nextRunning ? { configuration: bundle, controller: window.RLFlightGamepad?.getFrame() || null } : { elapsed: state.elapsed }
    }));
  });
  $('#stepButton').addEventListener('click', () => {
    if (state.running) return;
    try {
      const bundle = window.RLFlightConfig?.prepareSimulationStart();
      window.dispatchEvent(new CustomEvent('rlflightsimulationstep', {
        detail: { configuration: bundle, controller: window.RLFlightGamepad?.getFrame() || null }
      }));
    } catch (error) {
      const indicator = $('#dirtyIndicator');
      indicator.classList.add('dirty');
      indicator.querySelector('span').textContent = error.message;
    }
  });
  $('#resetButton').addEventListener('click', resetSimulation);
  $('#centerView').addEventListener('click', () => { state.cameraYaw = -.72; state.cameraPitch = .5; state.zoom = 1; state.topView = false; });
  $('#gridToggle').addEventListener('click', (event) => { state.grid = !state.grid; event.currentTarget.classList.toggle('active', state.grid); });
  $$('.view-button[data-view]').forEach(button => button.addEventListener('click', () => {
    $$('.view-button[data-view]').forEach(b => b.classList.remove('active'));
    button.classList.add('active'); state.topView = button.dataset.view === 'top';
  }));

  canvas.addEventListener('pointerdown', event => { state.dragging = true; state.dragX = event.clientX; state.dragY = event.clientY; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => {
    if (!state.dragging) return;
    state.topView = false; state.cameraYaw += (event.clientX - state.dragX) * .007; state.cameraPitch += (event.clientY - state.dragY) * .007;
    state.cameraPitch = Math.max(-1.2, Math.min(1.2, state.cameraPitch)); state.dragX = event.clientX; state.dragY = event.clientY;
  });
  canvas.addEventListener('pointerup', () => { state.dragging = false; });
  canvas.addEventListener('wheel', event => { event.preventDefault(); state.zoom = Math.max(.55, Math.min(1.8, state.zoom - event.deltaY * .0008)); }, { passive: false });

  function updateTargetLabels() {
    if (!$('#targetRoll')) return;
    const r = Number($('#targetRoll').value), p = Number($('#targetPitch').value), y = Number($('#targetYaw').value);
    setText('#targetRollOut', `${r.toFixed(1)}°`); setText('#targetPitchOut', `${p.toFixed(1)}°`); setText('#targetYawOut', `${y.toFixed(1)}°`);
    setText('#targetReadout', `R ${r >= 0 ? '+' : ''}${r.toFixed(1)}° \u00a0 P ${p >= 0 ? '+' : ''}${p.toFixed(1)}° \u00a0 Y ${y >= 0 ? '+' : ''}${y.toFixed(1)}°`);
  }

  function unwrapConfiguredValue(node, fallback) {
    if (node && typeof node === 'object' && !Array.isArray(node) && 'value' in node) return node.value;
    return node ?? fallback;
  }

  function applyAircraftGeometry(simConfig) {
    if (!simConfig?.aerodynamics || !simConfig?.body) return;
    aircraftVisual.geometry.centerOfMass = unwrapConfiguredValue(simConfig.body.center_of_mass_b, aircraftVisual.geometry.centerOfMass);
    aircraftVisual.geometry.directThrustCenter = unwrapConfiguredValue(simConfig.aerodynamics.direct_thrust_center_b, aircraftVisual.geometry.directThrustCenter);
    aircraftVisual.geometry.neutralThrustDirection = unwrapConfiguredValue(simConfig.aerodynamics.neutral_thrust_direction_b, aircraftVisual.geometry.neutralThrustDirection);
    if (Array.isArray(simConfig.aerodynamics.grids) && simConfig.aerodynamics.grids.length >= 3) {
      aircraftVisual.geometry.gridCenters = simConfig.aerodynamics.grids.slice(0, 3)
        .map((grid, index) => unwrapConfiguredValue(grid.aerodynamic_center_b, aircraftVisual.geometry.gridCenters[index]));
      aircraftVisual.geometry.gridAxes = simConfig.aerodynamics.grids.slice(0, 3)
        .map((grid, index) => unwrapConfiguredValue(grid.deflection_axis_b, aircraftVisual.geometry.gridAxes[index]));
    }
  }

  function firstBatch(value, expectedInnerLength) {
    if (Array.isArray(value) && Array.isArray(value[0]) && value.length === 1 && value[0].length === expectedInnerLength) return value[0];
    return value;
  }

  function validVector(value, length = 3) {
    return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
  }

  function quaternionToEuler(quaternion) {
    const [w, x, y, z] = quaternion;
    const sinr = 2 * (w * x + y * z);
    const cosr = 1 - 2 * (x * x + y * y);
    const sinp = Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
    const siny = 2 * (w * z + x * y);
    const cosy = 1 - 2 * (y * y + z * z);
    return [Math.atan2(sinr, cosr), Math.asin(sinp), Math.atan2(siny, cosy)];
  }

  function ingestAircraftTelemetry(payload) {
    const truth = payload?.truth || payload;
    if (!truth) return false;
    let gridMoments = firstBatch(truth.grid_moment_b, 3);
    let totalMoment = firstBatch(truth.moment_b, 3);
    if (Array.isArray(gridMoments) && gridMoments.length === 3 && gridMoments.every(vector => validVector(vector))) {
      aircraftVisual.gridMoments = gridMoments.map(vector => [...vector]);
    }
    if (validVector(totalMoment)) aircraftVisual.totalMoment = [...totalMoment];
    const servo = firstBatch(truth.servo_angle, 3);
    if (validVector(servo)) aircraftVisual.servoAngle = [...servo];
    const motors = firstBatch(truth.motor_speed, 2);
    if (validVector(motors, 2)) aircraftVisual.motorSpeed = [...motors];
    const quaternion = firstBatch(truth.attitude_q_wb, 4);
    if (validVector(quaternion, 4)) [state.roll, state.pitch, state.yaw] = quaternionToEuler(quaternion);
    const runtime = payload?.runtime;
    if (runtime && Number.isFinite(runtime.measured_control_hz)) {
      const measured = runtime.measured_control_hz;
      const requested = Math.max(1, Number(runtime.control_hz) || 1);
      $('#controlPerfBar').style.width = `${Math.min(100, measured / requested * 100)}%`;
      setText('#controlPerfValue', `${measured.toFixed(measured >= 100 ? 0 : 1)} Hz`);
      if (measured > 0) setText('#latencyValue', `${(1000 / measured).toFixed(2)} ms`);
    }
    if (payload.geometry) {
      const geometry = payload.geometry;
      if (validVector(geometry.center_of_mass_b)) aircraftVisual.geometry.centerOfMass = [...geometry.center_of_mass_b];
      if (validVector(geometry.direct_thrust_center_b)) aircraftVisual.geometry.directThrustCenter = [...geometry.direct_thrust_center_b];
      if (validVector(geometry.neutral_thrust_direction_b)) aircraftVisual.geometry.neutralThrustDirection = [...geometry.neutral_thrust_direction_b];
      if (Array.isArray(geometry.grid_aerodynamic_center_b) && geometry.grid_aerodynamic_center_b.every(vector => validVector(vector))) aircraftVisual.geometry.gridCenters = geometry.grid_aerodynamic_center_b.map(vector => [...vector]);
      if (Array.isArray(geometry.grid_deflection_axis_b) && geometry.grid_deflection_axis_b.every(vector => validVector(vector))) aircraftVisual.geometry.gridAxes = geometry.grid_deflection_axis_b.map(vector => [...vector]);
    }
    aircraftVisual.lastBackendAt = performance.now();
    return true;
  }

  const dirtyIndicator = $('#dirtyIndicator');
  $$('#parameters input, #parameters select, #parameters textarea').forEach(input => input.addEventListener('input', () => {
    dirtyIndicator.classList.add('dirty'); dirtyIndicator.querySelector('span').textContent = '有尚未应用的参数修改'; updateTargetLabels();
  }));
  $('#applyButton').addEventListener('click', () => {
    try {
      window.RLFlightConfig?.prepareSimulationStart();
    } catch (error) {
      dirtyIndicator.classList.add('dirty'); dirtyIndicator.querySelector('span').textContent = error.message;
    }
  });
  $('#restoreDefaults').addEventListener('click', () => {
    window.RLFlightConfig?.restoreDefaults(); updateTargetLabels(); dirtyIndicator.classList.add('dirty'); dirtyIndicator.querySelector('span').textContent = '默认参数待生成';
  });
  const appShell = $('.app-shell');
  const toggleLeftPanel = () => { appShell.classList.toggle('left-collapsed'); setTimeout(resizeCanvas, 220); };
  const toggleRightPanel = () => { appShell.classList.toggle('right-collapsed'); setTimeout(resizeCanvas, 220); };
  $('#collapseLeft').addEventListener('click', toggleLeftPanel);
  $('#toggleLeftPanel').addEventListener('click', toggleLeftPanel);
  $('#collapseRight').addEventListener('click', toggleRightPanel);
  $('#toggleRightPanel').addEventListener('click', toggleRightPanel);
  document.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') $('#applyButton').click(); });
  window.addEventListener('rlflightconfigurationready', event => {
    const simConfig = event.detail.simenv.config;
    applyAircraftGeometry(simConfig);
    const controlHz = simConfig.timing.control_hz.value;
    const physicsHz = simConfig.timing.physics_hz.value;
    $('#controlRateValue').innerHTML = `${controlHz} <small>Hz</small>`;
    $('#simRateValue').innerHTML = `${physicsHz >= 1000 ? (physicsHz / 1000).toFixed(1) : physicsHz} <small>${physicsHz >= 1000 ? 'kHz' : 'Hz'}</small>`;
  });
  window.addEventListener('rlflightconfigimported', event => {
    if (event.detail.kind === 'simenv') applyAircraftGeometry(event.detail.config);
  });
  window.addEventListener('rlflightsimulationtelemetry', event => ingestAircraftTelemetry(event.detail));
  window.addEventListener('rlflightruntimeerror', event => {
    setRunning(false);
    setText('#simStateLabel', 'FAULT');
    setText('#stateSymbol', '!');
    setText('#stateTitle', '运行时连接失败');
    setText('#stateHint', event.detail?.message || '远程 CPU 运行时错误');
  });
  window.RLFlightAircraft = {
    updateTelemetry: ingestAircraftTelemetry,
    updateGeometry: applyAircraftGeometry,
    getGeometry: () => JSON.parse(JSON.stringify(aircraftVisual.geometry))
  };
  setInterval(() => { $('#clock').textContent = new Date().toLocaleTimeString('en-GB', { hour12: false }); }, 1000);
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas(); updateTargetLabels(); requestAnimationFrame(frame);
})();
