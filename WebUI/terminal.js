(() => {
  'use strict';
  const modal = document.querySelector('#terminalModal');
  const output = document.querySelector('#terminalOutput');
  const input = document.querySelector('#terminalCommand');
  const append = (text) => {
    output.textContent += (output.textContent ? '\n' : '') + text;
    output.scrollTop = output.scrollHeight;
  };
  document.querySelector('#openTerminal')?.addEventListener('click', () => {
    modal.hidden = false;
    input.focus();
  });
  document.querySelector('#closeTerminal')?.addEventListener('click', () => { modal.hidden = true; });
  modal?.addEventListener('click', (event) => { if (event.target === modal) modal.hidden = true; });
  document.querySelector('#terminalForm')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const command = input.value.trim();
    if (!command) return;
    const runtime = window.RLFlightRuntime;
    let sessionId = runtime?.getSessionId?.();
    if (!sessionId) {
      try {
        const configuration = window.RLFlightConfig?.prepareSimulationStart?.();
        if (configuration?.test?.config?.runtime?.backend !== 'px4_hil') {
          append('[错误] 请先将 runtime.backend 设置为 px4_hil');
          return;
        }
        sessionId = await runtime.connect(configuration);
      } catch (error) {
        append(`[错误] 无法创建 px4_hil 会话：${error.message}`);
        return;
      }
    }
    input.value = '';
    append(`$ ${command}`);
    try {
      const response = await fetch(`/api/runtime/sessions/${sessionId}/terminal`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      append(data.output || '(no output)');
    } catch (error) { append(`[错误] ${error.message}`); }
    input.focus();
  });
})();
