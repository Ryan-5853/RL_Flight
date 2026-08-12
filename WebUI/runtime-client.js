(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  let sessionId = null;
  let latestControllerFrame = null;
  // Gamepad and virtual-input frames have independent source sequences.  The
  // HTTP protocol must use one monotonic transport sequence so switching input
  // sources can never make a live gamepad look older than a virtual frame.
  let transportSequence = 0;
  const controlInFlight = new Set();
  // Keep only two latest-value uploads in flight. More slots increase the age
  // of commands already queued in the browser without improving simulator
  // throughput.
  const MAX_CONTROL_IN_FLIGHT = 2;
  let lastTelemetrySequence = -1;
  let running = false;
  let creating = false;
  let controlTimer = null;
  let sessionKey = null;
  let failurePending = false;
  let virtualInputSequence = 0;
  let usingVirtualInput = false;
  let pendingTargetPosition = null;
  let telemetryAbortController = null;
  let clockOffsetMs = 0;
  let clockSyncRttMs = Number.NaN;
  const clientTraceBySequence = new Map();
  const epochNow = () => performance.timeOrigin + performance.now();

  function refreshVirtualInput() {
    usingVirtualInput = true;
    latestControllerFrame = {
      connected: true,
      sequence: ++virtualInputSequence,
      captured_epoch_ms: epochNow(),
      input_source: 'virtual',
      channels: { roll: 0, pitch: 0, yaw: 0, throttle: 0 }
    };
  }

  async function request(path, options = {}) {
    const requestStartedEpochMs = epochNow();
    const response = await fetch(path, {
      ...options,
      headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) }
    });
    const responseHeadersEpochMs = epochNow();
    const data = await response.json().catch(() => ({}));
    const responseParsedEpochMs = epochNow();
    if (data && typeof data === 'object') {
      Object.defineProperty(data, '__clientTiming', {
        configurable: true,
        enumerable: false,
        value: {
          request_started_epoch_ms: requestStartedEpochMs,
          response_headers_epoch_ms: responseHeadersEpochMs,
          response_parsed_epoch_ms: responseParsedEpochMs
        }
      });
    }
    if (!response.ok) throw new Error(data.error || `Runtime API HTTP ${response.status}`);
    return data;
  }

  async function synchronizeClock(sampleCount = 5) {
    const samples = [];
    for (let index = 0; index < sampleCount; index += 1) {
      try {
        const data = await request('/api/runtime/clock');
        const timing = data.__clientTiming;
        const t0 = timing.request_started_epoch_ms;
        const t3 = timing.response_parsed_epoch_ms;
        const t1 = Number(data.server_received_ns) / 1e6;
        const t2 = Number(data.server_response_started_ns) / 1e6;
        const networkRtt = Math.max(0, (t3 - t0) - Math.max(0, t2 - t1));
        const offset = ((t1 - t0) + (t2 - t3)) / 2;
        if (Number.isFinite(networkRtt) && Number.isFinite(offset)) {
          samples.push({ networkRtt, offset });
        }
      } catch (_) {
        break;
      }
    }
    if (!samples.length) return;
    samples.sort((a, b) => a.networkRtt - b.networkRtt);
    clockOffsetMs = samples[0].offset;
    clockSyncRttMs = samples[0].networkRtt;
  }

  function outgoingTrace(sequence) {
    const now = epochNow();
    const captured = Number(latestControllerFrame?.captured_epoch_ms);
    const trace = {
      transport_sequence: sequence,
      source_sequence: Number(latestControllerFrame?.sequence) || -1,
      input_source: latestControllerFrame?.input_source || (usingVirtualInput ? 'virtual' : 'gamepad'),
      input_captured_epoch_ms: Number.isFinite(captured) ? captured : now,
      hardware_updated_epoch_ms: Number(latestControllerFrame?.hardware_updated_epoch_ms) > 0
        ? Number(latestControllerFrame.hardware_updated_epoch_ms)
        : null,
      gamepad_poll_interval_ms: latestControllerFrame?.gamepad_poll_interval_ms != null
        && Number.isFinite(Number(latestControllerFrame.gamepad_poll_interval_ms))
        ? Number(latestControllerFrame.gamepad_poll_interval_ms)
        : null,
      client_send_epoch_ms: now
    };
    clientTraceBySequence.set(sequence, { ...trace });
    while (clientTraceBySequence.size > 256) {
      clientTraceBySequence.delete(clientTraceBySequence.keys().next().value);
    }
    return trace;
  }

  function recordControlAcknowledgement(sequence, data) {
    const trace = clientTraceBySequence.get(sequence);
    if (!trace || !data?.__clientTiming) return;
    trace.client_control_response_headers_epoch_ms = data.__clientTiming.response_headers_epoch_ms;
    trace.client_control_ack_epoch_ms = data.__clientTiming.response_parsed_epoch_ms;
  }

  function attachClientTelemetryTiming(telemetry, responseTiming) {
    if (!telemetry || !responseTiming) return telemetry;
    const trace = telemetry.latency_trace || (telemetry.latency_trace = {});
    const sequence = Number(trace.transport_sequence);
    const clientTrace = clientTraceBySequence.get(sequence);
    if (clientTrace) Object.assign(trace, clientTrace);
    trace.client_telemetry_headers_epoch_ms = responseTiming.response_headers_epoch_ms;
    trace.client_telemetry_parse_started_epoch_ms = responseTiming.parse_started_epoch_ms
      ?? responseTiming.response_headers_epoch_ms;
    trace.client_telemetry_parsed_epoch_ms = responseTiming.response_parsed_epoch_ms;
    trace.client_clock_offset_ms = clockOffsetMs;
    trace.client_clock_sync_rtt_ms = clockSyncRttMs;
    for (const key of [...clientTraceBySequence.keys()]) {
      if (key < sequence - 64) clientTraceBySequence.delete(key);
    }
    return telemetry;
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
    stopTelemetryStream();
    setConnection('CPU FAULT', 'fault');
    window.dispatchEvent(new CustomEvent('rlflightruntimeerror', { detail: { message: error.message || String(error) } }));
  }

  function configurationKey(configuration) {
    const test = JSON.parse(JSON.stringify(configuration.test.config));
    if (test.environment) test.environment.config_path = '<generated-simenv>';
    return JSON.stringify([configuration.simenv.config, test]);
  }

  async function ensureSession(configuration, controllerFrame = null) {
    if (controllerFrame?.connected) {
      latestControllerFrame = controllerFrame;
      usingVirtualInput = false;
    }
    if (!latestControllerFrame?.connected) refreshVirtualInput();
    const nextKey = configurationKey(configuration);
    if (sessionId && sessionKey !== nextKey) await closeSession();
    if (!sessionId) {
      const checkpointPath = configuration.test.config.runtime?.checkpoint_path;
      const data = await request('/api/runtime/sessions', {
        method: 'POST',
        body: JSON.stringify({
          simenv_yaml: configuration.simenv.yaml,
          test_yaml: configuration.test.yaml,
          checkpoint_path: checkpointPath,
          replace_existing: true
        })
      });
      sessionId = data.session.session_id;
      sessionKey = nextKey;
      transportSequence = 0;
      lastTelemetrySequence = -1;
      controlInFlight.clear();
      window.RLFlightLatency?.reset();
    }
    if (pendingTargetPosition) await sendPositionTarget();
    return sessionId;
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
      await ensureSession(configuration, controllerFrame);
      await sendControllerAction(singleStep ? 'step' : 'start');
      if (singleStep) {
        const data = await request(`/api/runtime/sessions/${sessionId}/telemetry?after=${lastTelemetrySequence}&timeout=1`);
        if (data.telemetry) {
          lastTelemetrySequence = data.telemetry.sequence;
          window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', {
            detail: attachClientTelemetryTiming(data.telemetry, data.__clientTiming)
          }));
        }
        setConnection('CPU PAUSED', 'ready');
        return;
      }
      running = true;
      setConnection('CPU RUNNING', 'active');
      startControlPump();
      receiveTelemetry(sessionId);
    } catch (error) {
      if (sessionId) await closeSession().catch(() => {});
      publishError(error);
    } finally {
      creating = false;
    }
  }

  async function applyConfiguration(detail) {
    if (creating) return;
    const configuration = detail?.configuration;
    if (!configuration) return;
    const resume = Boolean(detail?.resume);
    creating = true;
    setConnection('CPU RECONFIG', 'ready');
    try {
      if (sessionId) await closeSession().catch(() => {});
      sessionKey = null;
    } finally {
      creating = false;
    }
    if (resume) {
      await runSession(configuration, detail?.controller || null, false);
      if (!running) return;
    } else {
      setConnection('CPU READY', 'ready');
    }
    window.dispatchEvent(new CustomEvent('rlflightconfigurationapplied', {
      detail: {
        restarted: resume,
        sessionId
      }
    }));
  }

  function currentChannels() {
    if (usingVirtualInput || !latestControllerFrame?.connected) {
      refreshVirtualInput();
    }
    return { ...latestControllerFrame.channels };
  }

  async function sendPositionTarget(target = pendingTargetPosition) {
    if (!sessionId || !Array.isArray(target) || target.length !== 3) return;
    await request(`/api/runtime/sessions/${sessionId}/target`, {
      method: 'POST',
      body: JSON.stringify({ target_position_n: target })
    });
  }

  async function sendControllerAction(action) {
    if (!sessionId) return;
    if (controlInFlight.size) {
      await Promise.allSettled([...controlInFlight]);
    }
    const sequence = ++transportSequence;
    const trace = outgoingTrace(sequence);
    const pending = request(`/api/runtime/sessions/${sessionId}/${action}`, {
      method: 'POST',
      body: JSON.stringify({
        sequence,
        trace,
        channels: currentChannels()
      })
    });
    controlInFlight.add(pending);
    try {
      const data = await pending;
      if (!data.accepted) throw new Error(`controller frame ${sequence} was rejected as stale`);
      recordControlAcknowledgement(sequence, data);
    } finally {
      controlInFlight.delete(pending);
    }
  }

  async function sendControllerFrame() {
    if (!sessionId) return;
    if (controlInFlight.size >= MAX_CONTROL_IN_FLIGHT) return;
    const targetSession = sessionId;
    const channels = currentChannels();
    const sequence = ++transportSequence;
    const trace = outgoingTrace(sequence);
    const pending = request(`/api/runtime/sessions/${targetSession}/control`, {
      method: 'POST',
      body: JSON.stringify({ sequence, trace, channels })
    });
    controlInFlight.add(pending);
    try {
      const data = await pending;
      // Persistent connections may complete out of order. A rejected sequence
      // is simply older than a frame the runtime has already accepted.
      if (data.accepted) recordControlAcknowledgement(sequence, data);
    } finally {
      controlInFlight.delete(pending);
    }
  }

  function startControlPump() {
    clearInterval(controlTimer);
    // Gamepad rAF events send immediately. This low-rate timer is only a
    // watchdog keepalive for virtual input and browsers which temporarily
    // throttle animation callbacks.
    controlTimer = setInterval(() => {
      if (running) {
        if (usingVirtualInput) {
          latestControllerFrame.sequence = ++virtualInputSequence;
        }
        sendControllerFrame().catch(error => {
          if (running) failRuntime(error);
        });
      }
    }, 100);
  }

  function stopTelemetryStream() {
    telemetryAbortController?.abort();
    telemetryAbortController = null;
  }

  async function streamTelemetry(pollSession) {
    stopTelemetryStream();
    const abortController = new AbortController();
    telemetryAbortController = abortController;
    const response = await fetch(
      `/api/runtime/sessions/${pollSession}/telemetry-stream?after=${lastTelemetrySequence}`,
      {
        signal: abortController.signal,
        headers: { Accept: 'application/x-ndjson' }
      }
    );
    if (!response.ok || !response.body) {
      throw new Error(`telemetry stream HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = '';
    while (running && sessionId === pollSession) {
      const result = await reader.read();
      if (result.done) throw new Error('telemetry stream closed');
      const chunkReceivedEpochMs = epochNow();
      buffered += decoder.decode(result.value, { stream: true });
      const lines = buffered.split('\n');
      buffered = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const parseStartedEpochMs = epochNow();
        const data = JSON.parse(line);
        const parsedEpochMs = epochNow();
        if (data.telemetry && data.telemetry.sequence > lastTelemetrySequence) {
          lastTelemetrySequence = data.telemetry.sequence;
          window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', {
            detail: attachClientTelemetryTiming(data.telemetry, {
              response_headers_epoch_ms: chunkReceivedEpochMs,
              response_parsed_epoch_ms: parsedEpochMs,
              parse_started_epoch_ms: parseStartedEpochMs
            })
          }));
        }
        if (data.session?.state === 'faulted' || data.session?.fault) {
          throw new Error(data.session.fault || 'CPU runtime faulted');
        }
      }
    }
  }

  async function receiveTelemetry(pollSession) {
    try {
      await streamTelemetry(pollSession);
    } catch (error) {
      if (!running || sessionId !== pollSession || error?.name === 'AbortError') return;
      // Preserve the original long-poll endpoint as a compatibility fallback
      // for a browser or intermediary that cannot expose streaming bodies.
      await pollTelemetry(pollSession);
    }
  }

  async function pollTelemetry(pollSession) {
    while (running && sessionId === pollSession) {
      try {
        const data = await request(`/api/runtime/sessions/${pollSession}/telemetry?after=${lastTelemetrySequence}&timeout=1`);
        if (data.telemetry) {
          lastTelemetrySequence = data.telemetry.sequence;
          window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', {
            detail: attachClientTelemetryTiming(data.telemetry, data.__clientTiming)
          }));
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
    stopTelemetryStream();
    if (!sessionId) return;
    try {
      await request(`/api/runtime/sessions/${sessionId}/pause`, { method: 'POST', body: '{}' });
      setConnection('CPU PAUSED', 'ready');
    } catch (error) { publishError(error); }
  }

  async function resetSession() {
    pendingTargetPosition = null;
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
    stopTelemetryStream();
    controlInFlight.clear();
    if (closing) await request(`/api/runtime/sessions/${closing}`, { method: 'DELETE' });
  }

  function releaseSessionOnPageExit() {
    const closing = sessionId;
    if (!closing) return;
    sessionId = null;
    sessionKey = null;
    running = false;
    clearInterval(controlTimer);
    stopTelemetryStream();
    const endpoint = `/api/runtime/sessions/${closing}/close`;
    try {
      const body = new Blob(['{}'], { type: 'text/plain;charset=UTF-8' });
      if (navigator.sendBeacon?.(endpoint, body)) return;
    } catch (_) {
      // Fall through to keepalive fetch when Beacon is unavailable or denied.
    }
    fetch(endpoint, { method: 'POST', body: '{}', keepalive: true }).catch(() => {});
  }

  async function initializeRuntimeUi() {
    try {
      await synchronizeClock();
      const [capabilities, checkpointData] = await Promise.all([
        request('/api/runtime/capabilities'),
        request('/api/runtime/checkpoints')
      ]);
      const checkpointInput = document.querySelector('[data-config="test"][data-path="runtime.checkpoint_path"]');
      if (checkpointInput instanceof HTMLSelectElement) {
        checkpointInput.replaceChildren();
        const packages = checkpointData.checkpoints || [];
        const storedPackage = (() => {
          try { return localStorage.getItem('rlflight.inference-package.v1') || ''; }
          catch (_) { return ''; }
        })();
        const recommendedSimenv = new Map(
          packages
            .filter(item => item.recommended_simenv)
            .map(item => [item.path, item.recommended_simenv])
        );
        const applyRecommendedSimenv = () => {
          try { localStorage.setItem('rlflight.inference-package.v1', checkpointInput.value); }
          catch (_) { /* Selection remains valid for this page lifetime. */ }
          const config = recommendedSimenv.get(checkpointInput.value);
          if (!config || !window.RLFlightConfig?.importConfig) return;
          window.RLFlightConfig.importConfig(
            'simenv',
            config,
            `${checkpointInput.value} · 内置训练环境`
          );
        };
        if (!packages.length) {
          const option = document.createElement('option');
          option.value = '';
          option.textContent = '未发现推理包';
          checkpointInput.appendChild(option);
          checkpointInput.disabled = true;
        } else {
          packages.forEach(item => {
            const option = document.createElement('option');
            option.value = item.path;
            const family = item.contract_version === 'self_stabilize_v1'
              ? 'DIRECT'
              : item.contract_version === 'angular_acceleration_cascade_v1'
                ? 'CASCADE'
                : 'CUSTOM';
            option.textContent = `[${family}] ${item.path}`;
            checkpointInput.appendChild(option);
          });
          checkpointInput.disabled = false;
          const preferred = packages.find(item => item.path === storedPackage)
            || packages.find(item => item.contract_version === 'self_stabilize_v1')
            || packages[0];
          checkpointInput.value = preferred.path;
          checkpointInput.addEventListener('change', applyRecommendedSimenv);
          checkpointInput.dispatchEvent(new Event('input', { bubbles: true }));
          checkpointInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
      } else if (checkpointInput && checkpointData.checkpoints?.length) {
        const checkpoints = document.createElement('datalist');
        checkpoints.id = 'runtimeCheckpoints';
        checkpointData.checkpoints.forEach(item => {
          const option = document.createElement('option');
          option.value = item.path;
          checkpoints.appendChild(option);
        });
        document.body.appendChild(checkpoints);
        checkpointInput.setAttribute('list', checkpoints.id);
        if (!checkpointInput.value.trim()) {
          checkpointInput.value = checkpointData.checkpoints[0].path;
          checkpointInput.dispatchEvent(new Event('input', { bubbles: true }));
          checkpointInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
      setConnection(capabilities.cpu_available ? 'CPU READY' : 'CPU UNAVAILABLE', capabilities.cpu_available ? 'ready' : 'fault');
    } catch (_error) {
      setConnection('RUNTIME OFFLINE', 'fault');
    }
  }

  window.addEventListener('rlflightcontrollerframe', event => {
    if (event.detail?.connected) {
      latestControllerFrame = event.detail;
      usingVirtualInput = false;
    } else {
      refreshVirtualInput();
    }
    if (running) {
      sendControllerFrame().catch(error => {
        if (running) failRuntime(error);
      });
    }
  });
  window.addEventListener('rlflightsimulationstart', event => runSession(event.detail.configuration, event.detail.controller));
  window.addEventListener('rlflightsimulationstep', event => runSession(event.detail.configuration, event.detail.controller, true));
  window.addEventListener('rlflightsimulationpause', pauseSession);
  window.addEventListener('rlflightsimulationreset', resetSession);
  window.addEventListener('rlflightpositiontarget', event => {
    const target = event.detail?.target_position_n;
    if (!Array.isArray(target) || target.length !== 3 || !target.every(Number.isFinite)) return;
    pendingTargetPosition = [...target];
    if (sessionId) sendPositionTarget().catch(failRuntime);
  });
  window.addEventListener('rlflightsimulationtelemetry', event => {
    const reference = event.detail?.reference?.target_position_n;
    const target = Array.isArray(reference?.[0]) ? reference[0] : reference;
    if (Array.isArray(target) && target.length === 3 && target.every(Number.isFinite)) {
      pendingTargetPosition = [...target];
    }
  });
  window.addEventListener('rlflightconfigurationapply', event => {
    applyConfiguration(event.detail).catch(failRuntime);
  });
  // pagehide is delivered for refresh, tab close and history navigation more
  // reliably than beforeunload. The shared guard makes the fallback event safe.
  window.addEventListener('pagehide', releaseSessionOnPageExit);
  window.addEventListener('beforeunload', releaseSessionOnPageExit);

  window.RLFlightRuntime = {
    getSessionId: () => sessionId,
    getStatus: () => sessionId ? request(`/api/runtime/sessions/${sessionId}/status`) : Promise.resolve(null),
    getClockSync: () => ({ offsetMs: clockOffsetMs, rttMs: clockSyncRttMs }),
    close: closeSession
    ,connect: async configuration => {
      creating = true;
      try { return await ensureSession(configuration); }
      finally { creating = false; }
    }
  };
  initializeRuntimeUi();
})();
