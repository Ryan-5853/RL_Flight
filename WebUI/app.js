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
    positionNed: [0, 0, 0],
    targetPositionNed: [0, 0, 0],
    initialPositionNed: null,
    positionTrail: [],
    spatialEpisode: null,
    viewCenter: [0, 0, 0],
    viewHalfSpan: 4,
    gridStep: 1,
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
    gridForces: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    totalForce: [0, 0, 0],
    gridMoments: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    totalMoment: [0, 0, 0],
    motorSpeed: [0, 0],
    rotorPhase: [0, 0],
    servoAngle: [0, 0, 0],
    lastBackendAt: 0,
    backendSeen: false
  };
  let pendingLatencyTrace = null;
  const epochNow = () => performance.timeOrigin + performance.now();

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
  // NED/FRD 是右手系。Canvas 世界使用 NWU/FLU 右手系，因此必须同时翻转
  // East/Right 和 Down，不能只翻 Z 后落入左手坐标空间。
  const nedToWorld = ([north, east, down]) => [north, -east, -down];

  function vectorDistance(a, b) {
    return Math.hypot(
      (a[0] || 0) - (b[0] || 0),
      (a[1] || 0) - (b[1] || 0),
      (a[2] || 0) - (b[2] || 0)
    );
  }

  function niceGridStep(rawStep) {
    const safe = Math.max(0.01, Number(rawStep) || 1);
    const power = 10 ** Math.floor(Math.log10(safe));
    const fraction = safe / power;
    const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
    return niceFraction * power;
  }

  function formatGridValue(value, step = state.gridStep) {
    const decimals = step >= 1 ? 0 : Math.min(3, Math.max(1, Math.ceil(-Math.log10(step))));
    const normalized = Math.abs(value) < step * 1e-6 ? 0 : value;
    return normalized.toFixed(decimals);
  }

  function updateSpatialView() {
    const points = state.positionTrail.map(nedToWorld);
    const position = nedToWorld(state.positionNed);
    const target = nedToWorld(state.targetPositionNed);
    // 水平范围跟随本次轨迹，不强制包含可能很远的绝对坐标原点；
    // 垂直范围始终包含 NED D=0 平面，便于直观看到高度变化。
    points.push(position, target, [position[0], position[1], 0], [target[0], target[1], 0]);
    const minimum = [Infinity, Infinity, Infinity];
    const maximum = [-Infinity, -Infinity, -Infinity];
    points.forEach(point => point.forEach((value, axis) => {
      minimum[axis] = Math.min(minimum[axis], value);
      maximum[axis] = Math.max(maximum[axis], value);
    }));
    const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
    // 机体几何在画面中放大了四倍，因此边界至少额外保留 1.6 m。
    const requiredHalfSpan = Math.max(
      2.5,
      ...minimum.map((value, axis) => (maximum[axis] - value) / 2 + 1.6)
    );
    const step = niceGridStep(requiredHalfSpan / 4);
    state.viewCenter = center;
    state.viewHalfSpan = Math.max(2.5, Math.ceil(requiredHalfSpan / step) * step);
    state.gridStep = step;
    const badge = $('.view-badge b');
    if (badge) badge.textContent = `AUTO · ${formatGridValue(step, step)} m GRID`;
  }

  function resetSpatialVisualization(positionNed = null) {
    const position = validVector(positionNed) ? [...positionNed] : [0, 0, 0];
    state.positionNed = position;
    state.targetPositionNed = [...position];
    state.initialPositionNed = validVector(positionNed) ? [...positionNed] : null;
    state.positionTrail = validVector(positionNed) ? [[...positionNed]] : [];
    state.spatialEpisode = null;
    updateSpatialView();
  }

  function recordPosition(positionNed) {
    if (!validVector(positionNed)) return;
    const position = [...positionNed];
    if (!state.initialPositionNed) state.initialPositionNed = [...position];
    state.positionNed = position;
    const last = state.positionTrail[state.positionTrail.length - 1];
    if (!last || vectorDistance(last, position) >= 0.002) {
      state.positionTrail.push(position);
      // 长时间运行时对旧轨迹做均匀降采样，保留起点和最新点且限制绘制成本。
      if (state.positionTrail.length > 2400) {
        state.positionTrail = state.positionTrail.filter((_, index) => (
          index === 0 || index === state.positionTrail.length - 1 || index % 2 === 0
        ));
      }
    }
    updateSpatialView();
  }

  function worldProject(point) {
    const box = canvas.getBoundingClientRect();
    const relative = point.map((value, axis) => value - state.viewCenter[axis]);
    let p = state.topView ? rotateX(relative, Math.PI / 2) : rotateX(rotateZ(relative, state.cameraYaw), state.cameraPitch);
    const scale = Math.min(box.width, box.height) * 0.42 / state.viewHalfSpan * state.zoom;
    return [box.width / 2 + p[0] * scale, box.height * 0.54 - p[2] * scale - p[1] * scale * 0.08, p[1]];
  }

  function craftProject(point) {
    // SimEnv 机体系为 FRD，Canvas 机体系为 FLU；同时翻转 Right 和 Down
    // 可保持右手系。对应欧拉角为 Roll 保持、Pitch/Yaw 反号。
    let p = [point[0], -point[1], -point[2]];
    p = rotateX(p, state.roll);
    p = rotateY(p, -state.pitch);
    p = rotateZ(p, -state.yaw);
    const position = nedToWorld(state.positionNed);
    p = p.map((value, axis) => value + position[axis]);
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
    const majorStep = state.gridStep;
    const minorStep = majorStep / 2;
    const extent = state.viewHalfSpan * 1.45;
    const minX = Math.floor((state.viewCenter[0] - extent) / majorStep) * majorStep;
    const maxX = Math.ceil((state.viewCenter[0] + extent) / majorStep) * majorStep;
    const minY = Math.floor((state.viewCenter[1] - extent) / majorStep) * majorStep;
    const maxY = Math.ceil((state.viewCenter[1] + extent) / majorStep) * majorStep;
    for (let x = minX; x <= maxX + minorStep * .1; x += minorStep) {
      const major = Math.abs(x / majorStep - Math.round(x / majorStep)) < 1e-6;
      const axis = Math.abs(x) < minorStep * 1e-4;
      line(
        worldProject([x, minY, 0]),
        worldProject([x, maxY, 0]),
        axis ? '#6d393b' : major ? '#2a363c' : '#182126',
        axis ? 1.15 : major ? .8 : .5,
        axis ? .9 : major ? .8 : .62
      );
      if (major) {
        const label = worldProject([x, minY, 0]);
        ctx.fillStyle = axis ? '#cf6666' : '#607078';
        ctx.font = '500 7px IBM Plex Mono';
        ctx.textAlign = 'center';
        ctx.fillText(`N ${formatGridValue(x)}`, label[0], label[1] + 12);
      }
    }
    for (let y = minY; y <= maxY + minorStep * .1; y += minorStep) {
      const major = Math.abs(y / majorStep - Math.round(y / majorStep)) < 1e-6;
      const axis = Math.abs(y) < minorStep * 1e-4;
      line(
        worldProject([minX, y, 0]),
        worldProject([maxX, y, 0]),
        axis ? '#50613b' : major ? '#2a363c' : '#182126',
        axis ? 1.15 : major ? .8 : .5,
        axis ? .9 : major ? .8 : .62
      );
      if (major) {
        const label = worldProject([minX, y, 0]);
        ctx.fillStyle = axis ? '#9fbe68' : '#607078';
        ctx.font = '500 7px IBM Plex Mono';
        ctx.textAlign = 'right';
        ctx.fillText(`E ${formatGridValue(-y)}`, label[0] - 5, label[1] + 3);
      }
    }
    ctx.textAlign = 'left';
  }

  function drawAxis() {
    const origin = worldProject([0, 0, 0]);
    const length = state.gridStep * 2;
    const axes = [
      { p: [length, 0, 0], c: '#e66d6d', t: 'N / X' },
      { p: [0, -length, 0], c: '#b8dc74', t: 'E / Y' },
      { p: [0, 0, -length], c: '#6997f0', t: 'D / Z↓' }
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

  function drawVectorArrow(originRaw, vector, color, label, maxMagnitude, strong = false, dashed = false) {
    const mag = magnitude(vector);
    if (mag < 1e-7) return;
    const origin = originRaw.map(value => value * MODEL_SCALE);
    const relative = Math.min(1, mag / Math.max(maxMagnitude, 1e-9));
    const length = (strong ? .72 : .48) + relative * (strong ? .82 : .62);
    const direction = vector.map(value => value / mag);
    const end = origin.map((value, index) => value + direction[index] * length);
    const a = craftProject(origin), b = craftProject(end);
    if (dashed) ctx.setLineDash([4, 3]);
    line(a, b, color, strong ? 2.3 : 1.7, .95);
    if (dashed) ctx.setLineDash([]);
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const head = strong ? 9 : 7;
    ctx.beginPath(); ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - Math.cos(angle - .5) * head, b[1] - Math.sin(angle - .5) * head);
    ctx.lineTo(b[0] - Math.cos(angle + .5) * head, b[1] - Math.sin(angle + .5) * head);
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
    ctx.fillStyle = color; ctx.font = `${strong ? 600 : 500} 8px IBM Plex Mono`;
    ctx.fillText(`${label} ${mag.toFixed(3)}`, b[0] + 6, b[1] - 5);
  }

  function updateLoadFallback(now) {
    if (aircraftVisual.backendSeen) return;
    if (now - aircraftVisual.lastBackendAt < 600) return;
    if (!state.running) {
      aircraftVisual.gridForces = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      aircraftVisual.totalForce = [0, 0, 0];
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
    aircraftVisual.gridForces = [
      [0.18 * Math.sin(t), 0, -4.5],
      [-0.09 * Math.sin(t), 0.16 * Math.sin(t), -4.5],
      [-0.09 * Math.sin(t), -0.16 * Math.sin(t), -4.5]
    ];
    aircraftVisual.totalForce = aircraftVisual.gridForces
      .reduce((sum, item) => sum.map((value, axis) => value + item[axis]), [0, 0, -9]);
    aircraftVisual.totalMoment = aircraftVisual.gridMoments.reduce((sum, item) => sum.map((value, axis) => value + item[axis]), [0, 0, .006 * Math.sin(t)]);
    aircraftVisual.servoAngle = aircraftVisual.gridMoments.map(moment => Math.max(-.35, Math.min(.35, magnitude(moment) * 5)));
    aircraftVisual.motorSpeed = [1180 + Math.sin(t) * 60, 1120 + Math.cos(t * .8) * 55];
  }

  function drawLoads() {
    const gridForceMax = Math.max(
      ...aircraftVisual.gridForces.map(magnitude),
      1e-6
    );
    const colors = ['#55d8e6', '#b8dc74', '#eb9d61'];
    aircraftVisual.gridForces.forEach((force, index) => {
      drawVectorArrow(
        aircraftVisual.geometry.gridCenters[index],
        force,
        colors[index],
        `F${index + 1}`,
        gridForceMax
      );
      setText(`#force${index + 1}Value`, `${magnitude(force).toFixed(2)} N`);
    });
    drawVectorArrow(
      aircraftVisual.geometry.centerOfMass,
      aircraftVisual.totalForce,
      '#e8edf0',
      'ΣF',
      Math.max(magnitude(aircraftVisual.totalForce), 1e-6),
      true
    );
    drawVectorArrow(
      aircraftVisual.geometry.centerOfMass,
      aircraftVisual.totalMoment,
      '#b997e8',
      'ΣM axis',
      Math.max(magnitude(aircraftVisual.totalMoment), 1e-6),
      false,
      true
    );
    setText('#forceTotalValue', `${magnitude(aircraftVisual.totalForce).toFixed(2)} N`);
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
    drawLoads();
  }

  function drawTarget() {
    const target = nedToWorld(state.targetPositionNed);
    const radius = Math.max(.18, state.gridStep * .32);
    const center = worldProject(target);
    const ground = worldProject([target[0], target[1], 0]);
    ctx.setLineDash([3, 4]);
    line(ground, center, '#55d8e6', 1, .45);
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i <= 64; i++) {
      const p = worldProject([
        target[0] + Math.cos(i / 64 * Math.PI * 2) * radius,
        target[1] + Math.sin(i / 64 * Math.PI * 2) * radius,
        target[2]
      ]);
      i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
    }
    ctx.strokeStyle = '#2f484e'; ctx.setLineDash([4, 6]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(center[0], center[1], 3, 0, Math.PI * 2); ctx.fillStyle = '#55d8e6'; ctx.fill();
  }

  function drawPositionTrail() {
    const points = state.positionTrail.map(nedToWorld);
    if (!points.length) return;
    if (points.length > 1) {
      ctx.beginPath();
      points.forEach((point, index) => {
        const projected = worldProject(point);
        index ? ctx.lineTo(projected[0], projected[1]) : ctx.moveTo(projected[0], projected[1]);
      });
      ctx.strokeStyle = '#4ec7d4';
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = .82;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const start = worldProject(points[0]);
    ctx.beginPath();
    ctx.arc(start[0], start[1], 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#0b1114';
    ctx.fill();
    ctx.strokeStyle = '#55d8e6';
    ctx.lineWidth = 1;
    ctx.stroke();

    const current = nedToWorld(state.positionNed);
    const ground = worldProject([current[0], current[1], 0]);
    const aircraft = worldProject(current);
    ctx.setLineDash([3, 4]);
    line(ground, aircraft, '#6997f0', 1, .6);
    ctx.setLineDash([]);
  }

  function drawPositionReadout() {
    const position = state.positionNed;
    const initial = state.initialPositionNed || position;
    const displacement = vectorDistance(position, initial);
    const anchor = worldProject(nedToWorld(position));
    const text = `N ${position[0].toFixed(2)}  E ${position[1].toFixed(2)}  H ${(-position[2]).toFixed(2)} m  Δ ${displacement.toFixed(2)} m`;
    ctx.font = '600 8px IBM Plex Mono';
    const width = ctx.measureText(text).width + 14;
    const box = canvas.getBoundingClientRect();
    const x = Math.max(8, Math.min(box.width - width - 8, anchor[0] + 14));
    const y = Math.max(30, Math.min(box.height - 16, anchor[1] - 18));
    ctx.fillStyle = 'rgba(8, 14, 17, .82)';
    ctx.fillRect(x, y - 13, width, 19);
    ctx.strokeStyle = '#29444b';
    ctx.strokeRect(x + .5, y - 12.5, width - 1, 18);
    ctx.fillStyle = '#a9cbd0';
    ctx.textAlign = 'left';
    ctx.fillText(text, x + 7, y);
  }

  function drawScene() {
    const box = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, box.width, box.height);
    drawGrid();
    drawTarget();
    drawAxis();
    drawPositionTrail();
    drawCraft();
    drawPositionReadout();
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
    if (!aircraftVisual.backendSeen) {
      state.elapsed += dt;
      if (state.elapsed >= 30) {
        state.elapsed = 0;
        state.episode += 1;
      }
    }

    const targetR = Number($('#targetRoll')?.value || 0) * Math.PI / 180;
    const targetP = Number($('#targetPitch')?.value || 0) * Math.PI / 180;
    const targetY = Number($('#targetYaw')?.value || 0) * Math.PI / 180;
    if (!aircraftVisual.backendSeen && now - aircraftVisual.lastBackendAt >= 600) {
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
    if (!aircraftVisual.backendSeen && now - aircraftVisual.lastBackendAt >= 600) {
      const motorL = .62 + Math.sin(now / 420) * .035 - state.roll * .12;
      const motorR = .62 + Math.sin(now / 450) * .035 + state.roll * .12;
      setOutput('motorL', motorL, true); setOutput('motorR', motorR, true);
      setOutput('servoX', (targetR - state.roll) * 1.4); setOutput('servoY', (targetP - state.pitch) * 1.4); setOutput('servoZ', (targetY - state.yaw) * .7);
      setText('#angularVelocity', `${(targetR-state.roll).toFixed(2)}\u00a0\u00a0 ${(targetP-state.pitch).toFixed(2)}\u00a0\u00a0 ${(targetY-state.yaw).toFixed(2)} rad/s`);
      setText('#latencyValue', `${(0.31 + Math.sin(now / 800) * .04).toFixed(2)} ms`);
    }
    setText('#episodeNumber', String(state.episode).padStart(4, '0'));
    setText('#episodeTime', `00:${String(Math.floor(state.elapsed)).padStart(2, '0')}.${String(Math.floor(state.elapsed % 1 * 1000)).padStart(3, '0')} / 30s`);
    $('#episodeProgress').style.width = `${state.elapsed / 30 * 100}%`;
  }

  function frame(now) {
    const latencyTrace = pendingLatencyTrace;
    pendingLatencyTrace = null;
    if (latencyTrace) latencyTrace.frontend_render_started_epoch_ms = epochNow();
    updateTelemetry(now);
    updateLoadFallback(now);
    const visualDt = Math.min((now - state.lastFrame) / 1000, .05);
    aircraftVisual.rotorPhase[0] += aircraftVisual.motorSpeed[0] * visualDt * .035;
    aircraftVisual.rotorPhase[1] -= aircraftVisual.motorSpeed[1] * visualDt * .035;
    drawScene();
    if (latencyTrace) {
      latencyTrace.frontend_render_finished_epoch_ms = epochNow();
      window.dispatchEvent(new CustomEvent('rlflightlatencysample', {
        detail: latencyTrace
      }));
    }
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
    resetSpatialVisualization();
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
        indicator.classList.remove('dirty');
        indicator.classList.add('invalid');
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
      indicator.classList.remove('dirty');
      indicator.classList.add('invalid');
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
    const latencyTrace = payload?.latency_trace;
    if (latencyTrace) latencyTrace.frontend_apply_started_epoch_ms = epochNow();
    const truth = payload?.truth || payload;
    if (!truth) return false;
    let gridForces = firstBatch(truth.grid_force_b, 3);
    let totalForce = firstBatch(truth.force_b, 3);
    let gridMoments = firstBatch(truth.grid_moment_b, 3);
    let totalMoment = firstBatch(truth.moment_b, 3);
    if (Array.isArray(gridForces) && gridForces.length === 3 && gridForces.every(vector => validVector(vector))) {
      aircraftVisual.gridForces = gridForces.map(vector => [...vector]);
    }
    if (validVector(totalForce)) aircraftVisual.totalForce = [...totalForce];
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
    const angularVelocity = firstBatch(truth.angular_velocity_b, 3);
    if (validVector(angularVelocity)) {
      setText('#angularVelocity', `${angularVelocity.map(value => value.toFixed(3)).join('\u00a0\u00a0 ')} rad/s`);
    }
    const position = firstBatch(truth.position_n, 3);
    const runtime = payload?.runtime;
    const episodeId = Number(runtime?.episode_id);
    if (
      validVector(position)
      && Number.isFinite(episodeId)
      && state.spatialEpisode !== episodeId
    ) {
      if (state.spatialEpisode !== null) resetSpatialVisualization(position);
      state.spatialEpisode = episodeId;
    }
    if (validVector(position)) {
      recordPosition(position);
      const displacement = vectorDistance(position, state.initialPositionNed || position);
      setText('#positionValue', `N\u00a0 ${position[0].toFixed(3)}\u00a0\u00a0 E\u00a0 ${position[1].toFixed(3)}\u00a0\u00a0 D\u00a0 ${position[2].toFixed(3)} m\u00a0\u00a0 Δ\u00a0 ${displacement.toFixed(3)} m`);
    }
    const controllerCommand = firstBatch(payload?.controller?.command, 5);
    if (validVector(controllerCommand, 5)) {
      setOutput('motorL', controllerCommand[0], true);
      setOutput('motorR', controllerCommand[1], true);
      setOutput('servoX', controllerCommand[2]);
      setOutput('servoY', controllerCommand[3]);
      setOutput('servoZ', controllerCommand[4]);
    }
    const controllerType = payload?.controller?.type;
    if (controllerType) {
      const blendNode = payload.controller.diagnostics?.['controller.lqr_blend'];
      const blend = Array.isArray(blendNode) ? Number(blendNode[0]) : NaN;
      const suffix = Number.isFinite(blend) && controllerType === 'hybrid_pid_lqr'
        ? ` · LQR ${(blend * 100).toFixed(0)}%`
        : '';
      setText('#controllerMode', `${controllerType.toUpperCase()}${suffix}`);
      setText('#stateHint', `${controllerType.toUpperCase()} + SimEnv · CPU`);
    }
    const targetQuaternion = firstBatch(payload?.reference?.target_attitude_q_wb, 4);
    if (validVector(targetQuaternion, 4)) {
      const targetEuler = quaternionToEuler(targetQuaternion).map(value => value * 180 / Math.PI);
      setText('#targetReadout', `R ${targetEuler[0] >= 0 ? '+' : ''}${targetEuler[0].toFixed(1)}° \u00a0 P ${targetEuler[1] >= 0 ? '+' : ''}${targetEuler[1].toFixed(1)}° \u00a0 Y ${targetEuler[2] >= 0 ? '+' : ''}${targetEuler[2].toFixed(1)}°`);
    }
    const targetPosition = firstBatch(payload?.reference?.target_position_n, 3);
    if (validVector(targetPosition)) {
      state.targetPositionNed = [...targetPosition];
      updateSpatialView();
    }
    if (runtime && Number.isFinite(runtime.measured_control_hz)) {
      const effective = runtime.configuration;
      const centerOfMass = effective?.body?.center_of_mass_b;
      if (
        effective?.id
        && Array.isArray(centerOfMass)
        && centerOfMass.length === 3
        && centerOfMass.every(Number.isFinite)
      ) {
        const modelSource = effective.controller_parameter_source === 'environment'
          ? 'CTRL MODEL=ENV'
          : 'CTRL MODEL=CHECKPOINT';
        const effectiveNode = $('#effectiveConfig');
        effectiveNode.classList.add('active');
        effectiveNode.classList.remove('stale');
        effectiveNode.querySelector('span').textContent = `BACKEND ${effective.id} · COM [${centerOfMass.map(value => Number(value).toFixed(3)).join(', ')}] m · ${modelSource}`;
        effectiveNode.title = `FRD：x 向前、y 向右、z 正方向向下。后端实际质量 ${Number(effective.body.mass).toFixed(3)} kg。`;
      }
      const measured = runtime.measured_control_hz;
      const requested = Math.max(1, Number(runtime.control_hz) || 1);
      const realTimeFactor = Number.isFinite(runtime.real_time_factor)
        ? runtime.real_time_factor
        : measured / requested;
      $('#controlPerfBar').style.width = `${Math.min(100, measured / requested * 100)}%`;
      setText('#controlPerfValue', `${measured.toFixed(measured >= 100 ? 0 : 1)} Hz · ×${realTimeFactor.toFixed(2)}`);
      if (measured > 0) setText('#latencyValue', `${(1000 / measured).toFixed(2)} ms`);
      state.episode = Number(runtime.episode_id) || 0;
      state.elapsed = (Number(runtime.episode_step) || 0) / requested;
      const duration = Math.max(0.001, Number(document.querySelector('[data-path="task.episode_duration_s"]')?.value) || 30);
      setText('#episodeNumber', String(state.episode).padStart(4, '0'));
      setText('#episodeTime', `00:${String(Math.floor(state.elapsed)).padStart(2, '0')}.${String(Math.floor(state.elapsed % 1 * 1000)).padStart(3, '0')} / ${duration}s`);
      $('#episodeProgress').style.width = `${Math.min(100, state.elapsed / duration * 100)}%`;
      const reset = runtime.last_reset;
      if (reset && reset.reason && reset.reason !== 'manual') {
        const resetLabels = {
          simenv_invalid: `SimEnv 无效状态（错误码 ${reset.simenv_error_code}）`,
          tilt_limit: `倾角越界 ${(Number(reset.tilt_rad) * 180 / Math.PI).toFixed(1)}°`,
          angular_rate_limit: `角速度越界 ${Number(reset.angular_rate_rad_s).toFixed(2)} rad/s`,
          episode_timeout: 'Episode 正常到时'
        };
        setText('#stateHint', `最近重置：${resetLabels[reset.reason] || reset.reason} · step ${reset.episode_step}`);
      }
      if (runtime.controller_input?.stale) {
        setText(
          '#stateHint',
          `手柄帧暂时过期 · 中立安全输入 · 仿真继续（${runtime.controller_input.timeout_events} 次）`
        );
      }
    }
    if (payload.geometry) {
      const geometry = payload.geometry;
      if (validVector(geometry.center_of_mass_b)) aircraftVisual.geometry.centerOfMass = [...geometry.center_of_mass_b];
      if (validVector(geometry.direct_thrust_center_b)) aircraftVisual.geometry.directThrustCenter = [...geometry.direct_thrust_center_b];
      if (validVector(geometry.neutral_thrust_direction_b)) aircraftVisual.geometry.neutralThrustDirection = [...geometry.neutral_thrust_direction_b];
      if (Array.isArray(geometry.grid_aerodynamic_center_b) && geometry.grid_aerodynamic_center_b.every(vector => validVector(vector))) aircraftVisual.geometry.gridCenters = geometry.grid_aerodynamic_center_b.map(vector => [...vector]);
      if (Array.isArray(geometry.grid_deflection_axis_b) && geometry.grid_deflection_axis_b.every(vector => validVector(vector))) aircraftVisual.geometry.gridAxes = geometry.grid_deflection_axis_b.map(vector => [...vector]);
    }
    aircraftVisual.backendSeen = true;
    aircraftVisual.lastBackendAt = performance.now();
    if (latencyTrace) {
      latencyTrace.frontend_apply_finished_epoch_ms = epochNow();
      pendingLatencyTrace = latencyTrace;
    }
    return true;
  }

  const dirtyIndicator = $('#dirtyIndicator');
  $$('#parameters input, #parameters select, #parameters textarea').forEach(input => {
    input.addEventListener('input', updateTargetLabels);
  });
  $('#applyButton').addEventListener('click', () => {
    try {
      const bundle = window.RLFlightConfig?.prepareSimulationStart();
      window.dispatchEvent(new CustomEvent('rlflightconfigurationapply', {
        detail: {
          configuration: bundle,
          controller: window.RLFlightGamepad?.getFrame() || null,
          resume: state.running
        }
      }));
      if (state.running) {
        dirtyIndicator.querySelector('span').textContent = '正在用新参数重建仿真环境…';
      }
    } catch (error) {
      dirtyIndicator.classList.remove('dirty');
      dirtyIndicator.classList.add('invalid');
      dirtyIndicator.querySelector('span').textContent = error.message;
    }
  });
  $('#restoreDefaults').addEventListener('click', () => {
    window.RLFlightConfig?.restoreDefaults();
    updateTargetLabels();
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
  window.addEventListener('rlflightconfigurationapplied', event => {
    const indicator = $('#dirtyIndicator');
    indicator.classList.remove('dirty', 'invalid');
    indicator.querySelector('span').textContent = event.detail?.restarted
      ? '新参数已生效 · 仿真环境已重建'
      : '新参数已应用 · 下次启动将创建新环境';
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
    getGeometry: () => JSON.parse(JSON.stringify(aircraftVisual.geometry)),
    resetSpatialVisualization,
    getSpatialStatus() {
      const position = [...state.positionNed];
      const initial = state.initialPositionNed || position;
      return {
        position,
        displacement_m: vectorDistance(position, initial),
        grid_step_m: state.gridStep,
        view_span_m: state.viewHalfSpan * 2,
        trail_points: state.positionTrail.length,
        grid_visible: state.grid
      };
    },
    setGridVisible(visible) {
      state.grid = Boolean(visible);
      $('#gridToggle')?.classList.toggle('active', state.grid);
    }
  };
  setInterval(() => { $('#clock').textContent = new Date().toLocaleTimeString('en-GB', { hour12: false }); }, 1000);
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas(); updateSpatialView(); updateTargetLabels(); requestAnimationFrame(frame);
})();
