(() => {
  'use strict';

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  // v4 明确采用统一打杆语义：右杆向前/上为负 Pitch（低头），
  // Roll/Yaw 向右为正。硬件原始极性由校准向导负责识别。
  const STORAGE_KEY = 'rl-flight.gamepad.v4';
  // Mode 2 遥控器布局：左杆 X/Y 为偏航/油门，右杆 X/Y 为横滚/俯仰。
  const DEFAULT_MAPPING = { roll: 2, pitch: 3, yaw: 0, throttle: 1 };
  const DEFAULT_INVERTED = { roll: false, pitch: false, yaw: false, throttle: true };
  const CAPTURE_STEPS = [
    { channel: 'throttle', endpoint: 'positive', heading: '将左侧油门推到最大', description: '保持油门杆在最高位置，然后采集。' },
    { channel: 'throttle', endpoint: 'negative', heading: '将左侧油门拉到最小', description: '保持油门杆在最低位置，然后采集。' },
    { channel: 'yaw', endpoint: 'negative', heading: '将左侧偏航推到左极限', description: '保持左杆在最左位置，然后采集。' },
    { channel: 'yaw', endpoint: 'positive', heading: '将左侧偏航推到右极限', description: '保持左杆在最右位置，然后采集。' },
    { channel: 'pitch', endpoint: 'negative', heading: '将右侧俯仰向前推到上极限', description: '保持右杆在最上位置；该方向对应负 Pitch、机头下俯。' },
    { channel: 'pitch', endpoint: 'positive', heading: '将右侧俯仰向后拉到下极限', description: '保持右杆在最下位置；该方向对应正 Pitch、机头上仰。' },
    { channel: 'roll', endpoint: 'negative', heading: '将右侧横滚推到左极限', description: '保持右杆在最左位置，然后采集。' },
    { channel: 'roll', endpoint: 'positive', heading: '将右侧横滚推到右极限', description: '保持右杆在最右位置，然后采集。' }
  ];

  const controller = {
    activeIndex: null,
    gamepad: null,
    calibration: null,
    mapping: { ...DEFAULT_MAPPING },
    inverted: { ...DEFAULT_INVERTED },
    deadzone: 0.05,
    frame: emptyFrame(),
    wizardStep: 0,
    captures: {},
    calibrationBackup: null,
    lastPollEpochMs: null
  };

  function emptyFrame() {
    return { connected: false, id: null, timestamp: 0, sequence: 0, axes: [], buttons: [], channels: { roll: 0, pitch: 0, yaw: 0, throttle: 0 } };
  }

  function getPads() {
    if (!navigator.getGamepads) return [];
    return [...navigator.getGamepads()].filter(Boolean);
  }

  function storageId(gamepad) {
    return `${gamepad.id}::${gamepad.axes.length}::${gamepad.buttons.length}`;
  }

  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch { return {}; }
  }

  function saveProfile() {
    if (!controller.gamepad || !controller.calibration) return;
    const store = readStore();
    store[storageId(controller.gamepad)] = {
      calibration: controller.calibration,
      mapping: controller.mapping,
      inverted: controller.inverted,
      deadzone: controller.deadzone,
      savedAt: new Date().toISOString()
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  }

  function loadProfile(gamepad) {
    const profile = readStore()[storageId(gamepad)];
    controller.calibration = profile?.calibration || defaultCalibration(gamepad);
    controller.mapping = { ...DEFAULT_MAPPING, ...(profile?.mapping || {}) };
    controller.inverted = { ...DEFAULT_INVERTED, ...(profile?.inverted || {}) };
    controller.deadzone = Number.isFinite(profile?.deadzone) ? profile.deadzone : 0.05;
    $('#gamepadDeadzone').value = Math.round(controller.deadzone * 100);
    $('#deadzoneOut').textContent = `${Math.round(controller.deadzone * 100)}%`;
  }

  function defaultCalibration(gamepad) {
    return gamepad.axes.map(() => ({ min: -1, center: 0, max: 1 }));
  }

  function normalizeAxis(raw, calibration, deadzone) {
    const c = calibration || { min: -1, center: 0, max: 1 };
    const span = raw >= c.center ? c.max - c.center : c.center - c.min;
    let value = Math.abs(span) > 0.0001 ? (raw - c.center) / span : 0;
    value = Math.max(-1, Math.min(1, value));
    if (Math.abs(value) <= deadzone) return 0;
    return Math.sign(value) * (Math.abs(value) - deadzone) / (1 - deadzone);
  }

  function shortenId(id) {
    const clean = id.replace(/\s*\([^)]*\)\s*/g, ' ').replace(/\s+/g, ' ').trim();
    return clean.length > 32 ? `${clean.slice(0, 30)}…` : clean;
  }

  function selectGamepad(index) {
    const gamepad = getPads().find(pad => pad.index === Number(index));
    controller.activeIndex = gamepad?.index ?? null;
    controller.gamepad = gamepad || null;
    if (gamepad) loadProfile(gamepad);
    refreshDeviceUI();
  }

  function refreshDevices() {
    const pads = getPads();
    const select = $('#gamepadSelect');
    const previous = controller.activeIndex;
    select.innerHTML = '';
    if (!pads.length) {
      select.add(new Option('等待设备连接…', ''));
      select.disabled = true;
      controller.activeIndex = null;
      controller.gamepad = null;
      controller.frame = emptyFrame();
    } else {
      pads.forEach(pad => select.add(new Option(`${pad.index}: ${shortenId(pad.id)}`, String(pad.index))));
      select.disabled = false;
      const next = pads.some(pad => pad.index === previous) ? previous : pads[0].index;
      select.value = String(next);
      if (next !== controller.activeIndex || !controller.gamepad) selectGamepad(next);
    }
    refreshDeviceUI();
  }

  function populateMappings(axisCount) {
    $$('#mappingGrid select').forEach(select => {
      const channel = select.dataset.channel;
      select.innerHTML = '';
      for (let index = 0; index < axisCount; index += 1) select.add(new Option(`Axis ${index}`, String(index)));
      select.disabled = axisCount === 0;
      select.value = String(Math.min(controller.mapping[channel] ?? DEFAULT_MAPPING[channel], Math.max(0, axisCount - 1)));
      controller.mapping[channel] = Number(select.value || 0);
    });
    $$('[data-invert]').forEach(button => {
      button.disabled = axisCount === 0;
      button.classList.toggle('active', Boolean(controller.inverted[button.dataset.invert]));
    });
  }

  function refreshDeviceUI() {
    const connected = Boolean(controller.gamepad);
    const status = $('#gamepadDeviceStatus');
    status.classList.toggle('disconnected', !connected);
    status.querySelector('strong').textContent = connected ? shortenId(controller.gamepad.id) : '未检测到手柄';
    status.querySelector('small').textContent = connected
      ? `${controller.gamepad.mapping || 'raw'} · ${controller.gamepad.axes.length} axes · ${controller.gamepad.buttons.length} buttons`
      : '连接后按任意按键激活';
    $('#openCalibration').disabled = !connected;
    $('#gamepadTop').classList.toggle('disconnected', !connected);
    $('#gamepadTop span').textContent = connected ? 'GAMEPAD READY' : 'NO GAMEPAD';
    $('#gamepadMonitorState').textContent = connected ? 'CONNECTED' : 'DISCONNECTED';
    $('#gamepadMonitorState').classList.toggle('connected', connected);
    populateMappings(connected ? controller.gamepad.axes.length : 0);
  }

  function pollGamepad() {
    if (controller.activeIndex !== null) {
      const gamepad = getPads().find(pad => pad.index === controller.activeIndex);
      if (!gamepad) {
        controller.gamepad = null;
        refreshDevices();
      } else {
        controller.gamepad = gamepad;
        updateFrame(gamepad);
        if (!$('#calibrationModal').hidden) updateCalibrationUI(gamepad);
      }
    }
    requestAnimationFrame(pollGamepad);
  }

  function updateFrame(gamepad) {
    const capturedPerfMs = performance.now();
    const capturedEpochMs = performance.timeOrigin + capturedPerfMs;
    const pollIntervalMs = Number.isFinite(controller.lastPollEpochMs)
      ? Math.max(0, capturedEpochMs - controller.lastPollEpochMs)
      : null;
    controller.lastPollEpochMs = capturedEpochMs;
    const axes = gamepad.axes.map((value, index) => normalizeAxis(value, controller.calibration?.[index], controller.deadzone));
    const channels = {};
    Object.keys(DEFAULT_MAPPING).forEach(channel => {
      const value = axes[controller.mapping[channel]] ?? 0;
      channels[channel] = controller.inverted[channel] ? -value : value;
    });
    controller.frame = {
      connected: true,
      id: gamepad.id,
      index: gamepad.index,
      timestamp: capturedPerfMs,
      captured_epoch_ms: capturedEpochMs,
      gamepad_poll_interval_ms: pollIntervalMs,
      hardware_updated_epoch_ms: Number.isFinite(gamepad.timestamp) && gamepad.timestamp > 0
        ? performance.timeOrigin + gamepad.timestamp
        : null,
      input_source: 'gamepad',
      sequence: controller.frame.sequence + 1,
      axes,
      buttons: gamepad.buttons.map(button => ({ value: button.value, pressed: button.pressed, touched: button.touched })),
      mapping: { ...controller.mapping },
      inverted: { ...controller.inverted },
      channels
    };
    updateMonitor(channels);
    window.dispatchEvent(new CustomEvent('rlflightcontrollerframe', { detail: controller.frame }));
  }

  function updateMonitor({ roll, pitch, yaw, throttle }) {
    $('#leftStickDot').style.transform = `translate(${yaw * 18}px, ${-throttle * 18}px)`;
    // 负 Pitch 表示推杆低头，因此逻辑负值仍应把屏幕摇杆点画到上方。
    $('#rightStickDot').style.transform = `translate(${roll * 18}px, ${pitch * 18}px)`;
    const signed = value => `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
    $('#controllerAxesReadout').innerHTML = `R ${signed(roll)}&nbsp; P ${signed(pitch)}<br>Y ${signed(yaw)}&nbsp; T ${signed(throttle)}`;
  }

  function renderAxisRows(gamepad) {
    $('#axisCalibrationList').innerHTML = gamepad.axes.slice(0, 4).map((value, index) => `
      <div class="axis-calibration-item" data-axis="${index}">
        <div><span>AXIS ${index}</span><code>${value.toFixed(3)}</code></div>
        <div class="calibration-axis-track"><i style="left:${(value + 1) * 50}%"></i></div>
        <div class="calibration-axis-meta"><span>MIN <b>--</b></span><span>CENTER <b>--</b></span><span>MAX <b>--</b></span></div>
      </div>`).join('');
  }

  function updateCalibrationUI(gamepad) {
    gamepad.axes.slice(0, 4).forEach((value, index) => {
      const row = $(`.axis-calibration-item[data-axis="${index}"]`);
      if (!row) return;
      row.querySelector('code').textContent = value.toFixed(3);
      row.querySelector('.calibration-axis-track i').style.left = `${Math.max(0, Math.min(100, (value + 1) * 50))}%`;
      const labels = row.querySelectorAll('.calibration-axis-meta b');
      const axis = controller.calibration?.[index];
      labels[0].textContent = axis ? axis.min.toFixed(2) : '--';
      labels[1].textContent = axis ? axis.center.toFixed(2) : '--';
      labels[2].textContent = axis ? axis.max.toFixed(2) : '--';
      const channel = Object.keys(controller.mapping).find(name => controller.mapping[name] === index
        && controller.captures[`${name}:positive`] && controller.captures[`${name}:negative`]);
      row.querySelector('span').textContent = channel ? `${channel.toUpperCase()} · AXIS ${index}` : `AXIS ${index}`;
    });
  }

  function setWizardStep(step) {
    controller.wizardStep = step;
    const stage = step < CAPTURE_STEPS.length ? 0 : step === CAPTURE_STEPS.length ? 1 : 2;
    $$('[data-step-indicator]').forEach((item, index) => {
      item.classList.toggle('active', index === stage);
      item.classList.toggle('complete', index < stage);
    });
    let copy;
    if (step < CAPTURE_STEPS.length) {
      const action = CAPTURE_STEPS[step];
      copy = [`动作 ${step + 1} / 9`, action.heading, `${action.description} 系统会对当前位置连续采样 20 帧。`, '采集此位置', '只检查前四个轴；按钮、方向键和额外轴不会参与识别。'];
    } else if (step === CAPTURE_STEPS.length) {
      copy = ['动作 9 / 9', '释放横滚、俯仰和偏航摇杆', '让三个回中方向自然回到中心；油门位置不影响中心点采集。', '采集中心点', '中心点将连续采样 30 帧。油门使用最大与最小位置的中点，不要求物理回中。'];
    } else {
      copy = ['校准完成', '检查自动识别结果', '四个控制通道已经自动匹配轴号和方向，可以在右侧面板继续手动修正。', '保存校准', '只有 Roll、Pitch、Yaw、Throttle 使用的四个轴会被校准。按钮保持原始状态。'];
    }
    $('#calibrationKicker').textContent = copy[0];
    $('#calibrationHeading').textContent = copy[1];
    $('#calibrationDescription').textContent = copy[2];
    $('#calibrationNext').textContent = copy[3];
    $('#calibrationWarning span').textContent = copy[4];
  }

  async function sampleAxes(frameCount) {
    const samples = [];
    $('#calibrationNext').disabled = true;
    $('#calibrationNext').textContent = '采样中…';
    for (let frame = 0; frame < frameCount; frame += 1) {
      const gamepad = getPads().find(pad => pad.index === controller.activeIndex);
      if (!gamepad) break;
      samples.push([...gamepad.axes.slice(0, 4)]);
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    $('#calibrationNext').disabled = false;
    if (!samples.length) return null;
    return samples[0].map((_, index) => samples.reduce((sum, axes) => sum + axes[index], 0) / samples.length);
  }

  function identifyChannel(channel) {
    const positive = controller.captures[`${channel}:positive`];
    const negative = controller.captures[`${channel}:negative`];
    const usedAxes = new Set(Object.keys(controller.mapping)
      .filter(name => name !== channel && controller.captures[`${name}:positive`] && controller.captures[`${name}:negative`])
      .map(name => controller.mapping[name]));
    let bestAxis = -1;
    let bestTravel = 0;
    for (let index = 0; index < Math.min(4, positive.length); index += 1) {
      const travel = usedAxes.has(index) ? 0 : Math.abs(positive[index] - negative[index]);
      if (travel > bestTravel) { bestTravel = travel; bestAxis = index; }
    }
    if (bestAxis < 0 || bestTravel < 0.25) return false;
    controller.mapping[channel] = bestAxis;
    controller.inverted[channel] = positive[bestAxis] < negative[bestAxis];
    const min = Math.min(positive[bestAxis], negative[bestAxis]);
    const max = Math.max(positive[bestAxis], negative[bestAxis]);
    controller.calibration[bestAxis] = { min, center: (min + max) / 2, max };
    populateMappings(controller.gamepad.axes.length);
    return true;
  }

  async function captureGuidedPosition() {
    const step = CAPTURE_STEPS[controller.wizardStep];
    const values = await sampleAxes(20);
    if (!values) { closeCalibration(); return; }
    controller.captures[`${step.channel}:${step.endpoint}`] = values;
    const pairComplete = controller.captures[`${step.channel}:positive`] && controller.captures[`${step.channel}:negative`];
    if (pairComplete && !identifyChannel(step.channel)) {
      $('#calibrationWarning span').textContent = '没有检测到足够大的轴变化。请保持当前极限位置并重新采集，必要时返回重新开始校准。';
      $('#calibrationWarning').style.borderColor = '#633f32';
      return;
    }
    $('#calibrationWarning').style.borderColor = '';
    setWizardStep(controller.wizardStep + 1);
  }

  async function captureCenters() {
    const centers = await sampleAxes(30);
    if (!centers) { closeCalibration(); return; }
    ['roll', 'pitch', 'yaw'].forEach(channel => {
      const axisIndex = controller.mapping[channel];
      controller.calibration[axisIndex].center = centers[axisIndex];
    });
    const throttleAxis = controller.mapping.throttle;
    const throttle = controller.calibration[throttleAxis];
    throttle.center = (throttle.min + throttle.max) / 2;
    setWizardStep(CAPTURE_STEPS.length + 1);
  }

  function validateCalibration() {
    const mappedAxes = Object.values(controller.mapping);
    if (new Set(mappedAxes).size !== 4) return false;
    return mappedAxes.every(index => {
      const axis = controller.calibration[index];
      return axis && axis.max - axis.min > 0.25 && axis.center > axis.min && axis.center < axis.max;
    });
  }

  function saveCalibration() {
    if (!validateCalibration()) {
      $('#calibrationWarning span').textContent = '四个通道没有形成四根独立且有效的轴，请取消后重新校准。';
      $('#calibrationWarning').style.borderColor = '#633f32';
      return;
    }
    saveProfile();
    closeCalibration(true);
    $('#gamepadDeviceStatus small').textContent = '校准已保存 · 输入已标准化';
  }

  function openCalibration() {
    if (!controller.gamepad) return;
    if (controller.gamepad.axes.length < 4) {
      $('#gamepadDeviceStatus small').textContent = '至少需要四个轴才能进行遥控器校准';
      return;
    }
    controller.calibrationBackup = {
      calibration: controller.calibration.map(axis => ({ ...axis })),
      mapping: { ...controller.mapping },
      inverted: { ...controller.inverted }
    };
    $('#calibrationModal').hidden = false;
    renderAxisRows(controller.gamepad);
    controller.calibration = defaultCalibration(controller.gamepad);
    controller.captures = {};
    $('#calibrationWarning').style.borderColor = '';
    setWizardStep(0);
  }

  function closeCalibration(keepCalibration = false) {
    if (!keepCalibration && controller.calibrationBackup) {
      controller.calibration = controller.calibrationBackup.calibration.map(axis => ({ ...axis }));
      controller.mapping = { ...controller.calibrationBackup.mapping };
      controller.inverted = { ...controller.calibrationBackup.inverted };
      populateMappings(controller.gamepad?.axes.length || 0);
    }
    controller.calibrationBackup = null;
    $('#calibrationModal').hidden = true;
  }

  $('#gamepadSelect').addEventListener('change', event => selectGamepad(event.target.value));
  $$('#mappingGrid select').forEach(select => select.addEventListener('change', () => {
    controller.mapping[select.dataset.channel] = Number(select.value);
    saveProfile();
  }));
  $$('[data-invert]').forEach(button => button.addEventListener('click', () => {
    const channel = button.dataset.invert;
    controller.inverted[channel] = !controller.inverted[channel];
    button.classList.toggle('active', controller.inverted[channel]);
    saveProfile();
  }));
  $('#gamepadDeadzone').addEventListener('input', event => {
    controller.deadzone = Number(event.target.value) / 100;
    $('#deadzoneOut').textContent = `${event.target.value}%`;
  });
  $('#gamepadDeadzone').addEventListener('change', saveProfile);
  $('#openCalibration').addEventListener('click', openCalibration);
  $('#openCalibrationLeft').addEventListener('click', openCalibration);
  $('#gamepadTop').addEventListener('click', openCalibration);
  $('#closeCalibration').addEventListener('click', () => closeCalibration());
  $('#cancelCalibration').addEventListener('click', () => closeCalibration());
  $('#calibrationModal').addEventListener('click', event => { if (event.target === event.currentTarget) closeCalibration(); });
  $('#calibrationNext').addEventListener('click', () => {
    if (controller.wizardStep < CAPTURE_STEPS.length) captureGuidedPosition();
    else if (controller.wizardStep === CAPTURE_STEPS.length) captureCenters();
    else saveCalibration();
  });
  window.addEventListener('gamepadconnected', refreshDevices);
  window.addEventListener('gamepaddisconnected', refreshDevices);
  window.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('#calibrationModal').hidden) closeCalibration(); });

  const secureNote = $('#secureContextNote');
  if (window.isSecureContext) {
    secureNote.classList.add('secure');
    secureNote.querySelector('span').textContent = '当前为安全上下文，可读取 Windows 本机手柄。';
  }
  if (!navigator.getGamepads) {
    secureNote.querySelector('span').textContent = '当前浏览器不支持 Gamepad API，请使用最新版 Edge、Chrome 或 Firefox。';
  }

  window.RLFlightGamepad = {
    getFrame: () => typeof structuredClone === 'function' ? structuredClone(controller.frame) : JSON.parse(JSON.stringify(controller.frame)),
    getConnectedGamepads: getPads,
    openCalibration,
    storageKey: STORAGE_KEY
  };

  refreshDevices();
  requestAnimationFrame(pollGamepad);
})();
