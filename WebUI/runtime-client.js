(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  let sessionId = null;
  let latestControllerFrame = null;
  let lastSentSequence = -1;
  let lastTelemetrySequence = -1;
  let running = false;
  let creating = false;
  let controlTimer = null;
  let sessionKey = null;
  let failurePending = false;

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Runtime API HTTP ${response.status}`);
    return data;
  }

  function setConnection(label, state = 'ready') {
    const element = $('#runtimeConnection');
    element.querySelector('span').textContent = label;
    element.classList.toggle('runtime-fault', state === 'fault');
    element.classList.toggle('runtime-active', state === 'active');
  }

  function publishError(error) {
    running = false;
    clearInterval(controlTimer);
    setConnection('CPU FAULT', 'fault');
    window.dispatchEvent(new CustomEvent('rlflightruntimeerror', { detail: { message: error.message || String(error) } }));
  }

  function configurationKey(configuration) {
    const test = JSON.parse(JSON.stringify(configuration.test.config));
    if (test.environment) test.environment.config_path = '<generated-simenv>';
    return JSON.stringify([configuration.simenv.config, test]);
  }

  async function failRuntime(error) {
    if (failurePending) return;
    failurePending = true;
    running = false;
    clearInterval(controlTimer);
    if (sessionId) await closeSession().catch(() => {});
    publishError(error);
    failurePending = false;
  }

  async function runSession(configuration, controllerFrame = null, singleStep = false) {
    if (creating) return;
    creating = true;
    setConnection('CPU CONNECT', 'ready');
    try {
      if (controllerFrame?.connected) latestControllerFrame = controllerFrame;
      if (!latestControllerFrame?.connected) throw new Error('启动 CPU 会话前需要连接并激活手柄');
      const nextKey = configurationKey(configuration);
      if (sessionId && sessionKey !== nextKey) await closeSession();
      if (!sessionId) {
        const checkpointPath = configuration.test.config.runtime?.checkpoint_path;
        const data = await request('/api/runtime/sessions', {
          method: 'POST',
          body: JSON.stringify({
            simenv_yaml: configuration.simenv.yaml,
            test_yaml: configuration.test.yaml,
            checkpoint_path: checkpointPath
          })
        });
        sessionId = data.session.session_id;
        sessionKey = nextKey;
        lastSentSequence = -1;
        lastTelemetrySequence = -1;
      }
      await sendControllerFrame(true);
      await request(`/api/runtime/sessions/${sessionId}/${singleStep ? 'step' : 'start'}`, { method: 'POST', body: '{}' });
      if (singleStep) {
        const data = await request(`/api/runtime/sessions/${sessionId}/telemetry?after=${lastTelemetrySequence}&timeout=1`);
        if (data.telemetry) {
          lastTelemetrySequence = data.telemetry.sequence;
          window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', { detail: data.telemetry }));
        }
        setConnection('CPU PAUSED', 'ready');
        return;
      }
      running = true;
      setConnection('CPU RUNNING', 'active');
      startControlPump();
      pollTelemetry(sessionId);
    } catch (error) {
      if (sessionId) await closeSession().catch(() => {});
      publishError(error);
    } finally {
      creating = false;
    }
  }

  async function sendControllerFrame(force = false) {
    if (!sessionId || !latestControllerFrame?.connected) return;
    if (!force && latestControllerFrame.sequence <= lastSentSequence) return;
    const sequence = latestControllerFrame.sequence;
    const data = await request(`/api/runtime/sessions/${sessionId}/control`, {
      method: 'POST',
      body: JSON.stringify({ sequence, client_time_ms: performance.now(), channels: latestControllerFrame.channels })
    });
    if (data.accepted) lastSentSequence = sequence;
  }

  function startControlPump() {
    clearInterval(controlTimer);
    // HTTP is only the human-input boundary. The server holds the latest command
    // while its independent CPU loop continues at control_hz.
    controlTimer = setInterval(() => {
      if (running) sendControllerFrame().catch(failRuntime);
    }, 16);
  }

  async function pollTelemetry(pollSession) {
    while (running && sessionId === pollSession) {
      try {
        const data = await request(`/api/runtime/sessions/${pollSession}/telemetry?after=${lastTelemetrySequence}&timeout=1`);
        if (data.telemetry) {
          lastTelemetrySequence = data.telemetry.sequence;
          window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', { detail: data.telemetry }));
        }
        if (data.session.state === 'faulted' || data.session.fault) throw new Error(data.session.fault || 'CPU runtime faulted');
      } catch (error) {
        if (running && sessionId === pollSession) await failRuntime(error);
      }
    }
  }

  async function pauseSession() {
    running = false;
    clearInterval(controlTimer);
    if (!sessionId) return;
    try {
      await request(`/api/runtime/sessions/${sessionId}/pause`, { method: 'POST', body: '{}' });
      setConnection('CPU PAUSED', 'ready');
    } catch (error) { publishError(error); }
  }

  async function resetSession() {
    if (!sessionId) return;
    try { await request(`/api/runtime/sessions/${sessionId}/reset`, { method: 'POST', body: '{}' }); }
    catch (error) { publishError(error); }
  }

  async function closeSession() {
    const closing = sessionId;
    sessionId = null;
    sessionKey = null;
    running = false;
    clearInterval(controlTimer);
    if (closing) await request(`/api/runtime/sessions/${closing}`, { method: 'DELETE' });
  }

  async function initializeRuntimeUi() {
    try {
      const [capabilities, checkpointData] = await Promise.all([
        request('/api/runtime/capabilities'),
        request('/api/runtime/checkpoints')
      ]);
      const checkpointInput = document.querySelector('[data-config="test"][data-path="runtime.checkpoint_path"]');
      if (checkpointInput && checkpointData.checkpoints?.length) {
        const checkpoints = document.createElement('datalist');
        checkpoints.id = 'runtimeCheckpoints';
        checkpointData.checkpoints.forEach(item => {
          const option = document.createElement('option');
          option.value = item.path;
          checkpoints.appendChild(option);
        });
        document.body.appendChild(checkpoints);
        checkpointInput.setAttribute('list', checkpoints.id);
      }
      setConnection(capabilities.cpu_available ? 'CPU READY' : 'CPU UNAVAILABLE', capabilities.cpu_available ? 'ready' : 'fault');
    } catch (_error) {
      setConnection('RUNTIME OFFLINE', 'fault');
    }
  }

  window.addEventListener('rlflightcontrollerframe', event => { latestControllerFrame = event.detail; });
  window.addEventListener('rlflightsimulationstart', event => runSession(event.detail.configuration, event.detail.controller));
  window.addEventListener('rlflightsimulationstep', event => runSession(event.detail.configuration, event.detail.controller, true));
  window.addEventListener('rlflightsimulationpause', pauseSession);
  window.addEventListener('rlflightsimulationreset', resetSession);
  window.addEventListener('beforeunload', () => {
    running = false;
    clearInterval(controlTimer);
    if (sessionId) fetch(`/api/runtime/sessions/${sessionId}`, { method: 'DELETE', keepalive: true }).catch(() => {});
  });

  window.RLFlightRuntime = {
    getSessionId: () => sessionId,
    getStatus: () => sessionId ? request(`/api/runtime/sessions/${sessionId}/status`) : Promise.resolve(null),
    close: closeSession
  };
  initializeRuntimeUi();
})();
