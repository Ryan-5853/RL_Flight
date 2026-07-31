(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const WINDOW_SIZE = 240;
  const histories = new Map();
  let sampleCount = 0;

  const serverEpochMs = (trace, key) => {
    if (trace[key] === null || trace[key] === undefined) return Number.NaN;
    const value = Number(trace[key]);
    const offset = Number(trace.client_clock_offset_ms) || 0;
    return Number.isFinite(value) ? value / 1e6 - offset : Number.NaN;
  };
  const clientEpochMs = (trace, key) => {
    if (trace[key] === null || trace[key] === undefined) return Number.NaN;
    const value = Number(trace[key]);
    return Number.isFinite(value) ? value : Number.NaN;
  };
  const duration = (end, start) => (
    Number.isFinite(end) && Number.isFinite(start) ? Math.max(0, end - start) : Number.NaN
  );
  const serverDuration = (trace, end, start) => duration(
    serverEpochMs(trace, end),
    serverEpochMs(trace, start)
  );
  const clientDuration = (trace, end, start) => duration(
    clientEpochMs(trace, end),
    clientEpochMs(trace, start)
  );
  const measuredNs = (trace, key) => {
    if (trace[key] === null || trace[key] === undefined) return Number.NaN;
    const value = Number(trace[key]);
    return Number.isFinite(value) ? Math.max(0, value / 1e6) : Number.NaN;
  };

  const stages = [
    {
      id: 'total',
      label: '整链路总延迟',
      group: 'total',
      title: '手柄/虚拟输入采样完成，到对应状态完成 Canvas 绘制提交',
      value: trace => clientDuration(trace, 'frontend_render_finished_epoch_ms', 'input_captured_epoch_ms')
    },
    {
      id: 'gamepad_poll',
      label: '手柄轮询周期',
      group: 'frontend',
      title: '浏览器相邻两次 Gamepad API 读取之间的周期；这是采样频率诊断项，与主链路并行，不重复计入整链路总延迟',
      value: trace => {
        if (trace.gamepad_poll_interval_ms == null) return Number.NaN;
        const value = Number(trace.gamepad_poll_interval_ms);
        return Number.isFinite(value) && value >= 0 ? value : Number.NaN;
      }
    },
    {
      id: 'input_queue',
      label: '输入采样 → 发送',
      group: 'frontend',
      title: '浏览器读取手柄后，等待 40 Hz 控制发送泵的时间',
      value: trace => clientDuration(trace, 'client_send_epoch_ms', 'input_captured_epoch_ms')
    },
    {
      id: 'uplink',
      label: '控制指令上行',
      group: 'network',
      title: '前端发起控制请求，到后端 HTTP 入口收到请求；依赖客户端/服务器时钟同步',
      value: trace => duration(
        serverEpochMs(trace, 'server_control_received_ns'),
        clientEpochMs(trace, 'client_send_epoch_ms')
      )
    },
    {
      id: 'server_input',
      label: 'HTTP 解析与入队',
      group: 'backend',
      title: '后端收到 HTTP 请求，到控制帧写入运行时会话',
      value: trace => serverDuration(trace, 'command_applied_ns', 'server_control_received_ns')
    },
    {
      id: 'scheduler',
      label: '等待执行周期',
      group: 'backend',
      title: '控制帧入队后，等待后端仿真线程开始使用该帧',
      value: trace => serverDuration(trace, 'backend_step_started_ns', 'command_applied_ns')
    },
    {
      id: 'preparation',
      label: '观测与目标构造',
      group: 'backend',
      title: '输入映射、SimEnv 观测读取、状态与控制目标构造',
      value: trace => serverDuration(trace, 'controller_started_ns', 'backend_step_started_ns')
    },
    {
      id: 'controller',
      label: '控制器推理',
      group: 'backend',
      title: 'PID、LQR、混合控制器或神经网络前向推理',
      value: trace => measuredNs(trace, 'controller_elapsed_ns')
    },
    {
      id: 'environment',
      label: '环境动力学仿真',
      group: 'backend',
      title: 'SimEnv environment.advance 完整执行时间',
      value: trace => measuredNs(trace, 'environment_elapsed_ns')
    },
    {
      id: 'post_step',
      label: '安全检查与步后处理',
      group: 'backend',
      title: '仿真推进之后的终止判断、复位检查和状态提交',
      value: trace => serverDuration(trace, 'post_step_finished_ns', 'environment_finished_ns')
    },
    {
      id: 'telemetry_pack',
      label: '遥测采集与封装',
      group: 'backend',
      title: '读取可视化真值、控制诊断并构造遥测数据包',
      value: trace => serverDuration(trace, 'telemetry_pack_finished_ns', 'telemetry_pack_started_ns')
    },
    {
      id: 'telemetry_wait',
      label: '遥测等待响应',
      group: 'backend',
      title: '遥测数据生成后，等待流式发送线程取走并开始响应',
      value: trace => serverDuration(trace, 'server_response_started_ns', 'telemetry_published_ns')
    },
    {
      id: 'downlink',
      label: '序列化与参数下行',
      group: 'network',
      title: '后端开始发送该遥测帧，到浏览器收到流式数据块；包含 JSON 序列化、Socket 发送与网络下行',
      value: trace => duration(
        clientEpochMs(trace, 'client_telemetry_headers_epoch_ms'),
        serverEpochMs(trace, 'server_response_started_ns')
      )
    },
    {
      id: 'json_parse',
      label: '响应接收与 JSON 解析',
      group: 'frontend',
      title: '浏览器收到完整流式数据块，到解析完成遥测 JSON',
      value: trace => clientDuration(trace, 'client_telemetry_parsed_epoch_ms', 'client_telemetry_parse_started_epoch_ms')
    },
    {
      id: 'frontend_apply',
      label: '前端参数应用',
      group: 'frontend',
      title: '将遥测参数写入姿态、输出、目标和 DOM 状态',
      value: trace => clientDuration(trace, 'frontend_apply_finished_epoch_ms', 'frontend_apply_started_epoch_ms')
    },
    {
      id: 'render_wait',
      label: '等待下一渲染帧',
      group: 'frontend',
      title: '参数应用完成后，等待浏览器下一次 requestAnimationFrame',
      value: trace => clientDuration(trace, 'frontend_render_started_epoch_ms', 'frontend_apply_finished_epoch_ms')
    },
    {
      id: 'canvas_render',
      label: 'Canvas 绘制提交',
      group: 'frontend',
      title: '当前帧状态更新与 Canvas 绘制命令执行时间，不包含显示器扫描输出',
      value: trace => clientDuration(trace, 'frontend_render_finished_epoch_ms', 'frontend_render_started_epoch_ms')
    },
    {
      id: 'control_rtt',
      label: '控制请求 RTT',
      group: 'network',
      title: '控制请求发出到后端确认被浏览器解析完成；此项与主链路并行，不计入阶段求和',
      value: trace => clientDuration(trace, 'client_control_ack_epoch_ms', 'client_send_epoch_ms')
    }
  ];

  function percentile(values, fraction) {
    if (!values.length) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
  }

  function statsFor(id, current) {
    if (Number.isFinite(current) && current < 60000) {
      const history = histories.get(id) || [];
      history.push(current);
      if (history.length > WINDOW_SIZE) history.shift();
      histories.set(id, history);
    }
    const history = histories.get(id) || [];
    return {
      current,
      mean: history.length ? history.reduce((sum, value) => sum + value, 0) / history.length : Number.NaN,
      p95: percentile(history, .95),
      max: history.length ? Math.max(...history) : Number.NaN,
      count: history.length
    };
  }

  function formatMs(value) {
    if (!Number.isFinite(value)) return '--';
    if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  function renderTableShell() {
    const body = $('#latencyTableBody');
    if (!body) return;
    body.innerHTML = stages.map(stage => `
      <tr class="${stage.group}" data-latency-row="${stage.id}" title="${stage.title}">
        <td>${stage.label}</td>
        <td data-stat="current">--</td>
        <td data-stat="mean">--</td>
        <td data-stat="p95">--</td>
      </tr>`).join('');
  }

  function ingest(trace) {
    if (!trace || typeof trace !== 'object') return;
    sampleCount += 1;
    const snapshot = {};
    stages.forEach(stage => {
      const stats = statsFor(stage.id, stage.value(trace));
      snapshot[stage.id] = stats;
      const row = document.querySelector(`[data-latency-row="${stage.id}"]`);
      if (!row) return;
      row.classList.toggle('missing', !Number.isFinite(stats.current));
      ['current', 'mean', 'p95'].forEach(key => {
        row.querySelector(`[data-stat="${key}"]`).textContent = formatMs(stats[key]);
      });
    });

    const total = snapshot.total;
    $('#latencyTotalCurrent').innerHTML = `${formatMs(total.current)}<small>${total.current >= 1000 ? '' : ' ms'}</small>`;
    $('#latencyTotalP95').innerHTML = `${formatMs(total.p95)}<small>${total.p95 >= 1000 ? '' : ' ms'}</small>`;
    const state = $('#latencyTraceState');
    state.textContent = trace.input_stale ? 'FAILSAFE' : 'TRACING';
    state.classList.toggle('active', !trace.input_stale);
    const source = trace.input_source === 'gamepad' ? 'GAMEPAD' : trace.input_source === 'virtual' ? 'VIRTUAL' : String(trace.input_source || 'UNKNOWN').toUpperCase();
    const transport = trace.telemetry_transport === 'stream' ? 'STREAM' : trace.telemetry_transport === 'long_poll' ? 'POLL' : 'LINK';
    $('#latencySampleMeta').textContent = `${source} · ${transport} · CTRL #${trace.transport_sequence ?? '--'} · 样本 ${sampleCount} / 窗口 ${Math.min(sampleCount, WINDOW_SIZE)}`;
    const syncRtt = Number(trace.client_clock_sync_rtt_ms);
    const uncertainty = Number.isFinite(syncRtt) ? syncRtt / 2 : Number.NaN;
    $('#latencyClockMeta').textContent = Number.isFinite(uncertainty)
      ? `CLOCK SYNC ±${formatMs(uncertainty)} ms · MAX E2E ${formatMs(total.max)} ms`
      : `CLOCK UNSYNCED · MAX E2E ${formatMs(total.max)} ms`;
    const backendStep = serverDuration(trace, 'backend_step_finished_ns', 'backend_step_started_ns');
    if (Number.isFinite(backendStep)) {
      $('#latencyValue').innerHTML = `${formatMs(backendStep)} <small>ms</small>`;
    }
    window.dispatchEvent(new CustomEvent('rlflightlatencystats', {
      detail: { trace, stages: snapshot, sampleCount }
    }));
  }

  renderTableShell();
  window.addEventListener('rlflightlatencysample', event => ingest(event.detail));
  window.RLFlightLatency = {
    ingest,
    reset: () => {
      histories.clear();
      sampleCount = 0;
      renderTableShell();
    },
    getSnapshot: () => Object.fromEntries(
      [...histories.entries()].map(([key, values]) => [key, [...values]])
    )
  };
})();
