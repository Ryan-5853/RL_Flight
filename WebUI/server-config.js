(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const modal = $('#serverConfigModal');
  const rootSelect = $('#serverConfigRoot');
  const fileList = $('#serverConfigFiles');
  const status = $('#serverConfigStatus');
  let currentPath = '';
  let selectedFile = null;

  async function request(path, params = {}) {
    const query = new URLSearchParams(params);
    const response = await fetch(`${path}${query.size ? `?${query}` : ''}`, { headers: { Accept: 'application/json' } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `服务器返回 HTTP ${response.status}`);
    return data;
  }

  function setStatus(message, error = false) {
    status.classList.toggle('error', error);
    status.querySelector('span').textContent = message;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  }

  async function loadRoots() {
    setStatus('正在读取服务器允许的配置目录…');
    const data = await request('/api/config/roots');
    rootSelect.innerHTML = '';
    data.roots.forEach(root => {
      const option = new Option(`${root.label} · ${root.path}`, root.id);
      option.title = root.path;
      rootSelect.add(option);
    });
    if (!data.roots.length) throw new Error('服务器没有配置任何可访问的配置根目录。');
    currentPath = '';
    await loadDirectory();
  }

  async function loadDirectory(path = currentPath) {
    selectedFile = null;
    $('#importServerConfig').disabled = true;
    setStatus('正在读取服务器目录…');
    const data = await request('/api/config/files', { root: rootSelect.value, path });
    currentPath = data.path === '.' ? '' : data.path;
    $('#serverConfigPath').textContent = `server://${rootSelect.value}/${currentPath}`;
    fileList.innerHTML = '';
    if (!data.entries.length) {
      fileList.innerHTML = '<div class="server-empty">此服务器目录中没有 YAML 或 JSON 配置</div>';
    }
    data.entries.forEach(entry => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `server-file ${entry.kind}`;
      button.dataset.path = entry.path;
      button.dataset.kind = entry.kind;
      const icon = document.createElement('i');
      icon.textContent = entry.kind === 'directory' ? '▸' : 'YML';
      const copy = document.createElement('span');
      const name = document.createElement('strong');
      const meta = document.createElement('small');
      name.textContent = entry.name;
      meta.textContent = entry.kind === 'directory' ? '服务器目录' : formatSize(entry.size);
      copy.append(name, meta);
      button.append(icon, copy);
      button.addEventListener('click', () => selectEntry(button, entry));
      button.addEventListener('dblclick', () => entry.kind === 'directory' ? loadDirectory(entry.path) : importSelected());
      fileList.appendChild(button);
    });
    setStatus(`${data.entries.length} 个项目 · 仅显示服务器上的 YAML/JSON`);
  }

  function selectEntry(button, entry) {
    if (entry.kind === 'directory') {
      loadDirectory(entry.path).catch(showError);
      return;
    }
    document.querySelectorAll('.server-file.selected').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
    selectedFile = entry;
    $('#importServerConfig').disabled = false;
    setStatus(`已选择 ${entry.name}`);
  }

  async function importSelected() {
    if (!selectedFile) return;
    $('#importServerConfig').disabled = true;
    setStatus(`正在从服务器读取 ${selectedFile.name}…`);
    try {
      const data = await request('/api/config/file', { root: rootSelect.value, path: selectedFile.path });
      const result = window.RLFlightConfig.importConfig(data.kind, data.config, data.path);
      closeModal();
      setStatus(`文件覆盖 ${result.applied} 项 · 默认补齐 ${result.defaulted} 项`);
    } catch (error) {
      showError(error);
      $('#importServerConfig').disabled = false;
    }
  }

  function showError(error) { setStatus(error.message || String(error), true); }
  function closeModal() { modal.hidden = true; }

  async function openModal() {
    modal.hidden = false;
    selectedFile = null;
    $('#importServerConfig').disabled = true;
    try { await loadRoots(); }
    catch (error) {
      showError(error.message.includes('Failed to fetch')
        ? new Error('无法连接服务器配置接口。请使用 server.py 启动 WebUI，而不是普通静态文件服务器。')
        : error);
    }
  }

  $('#openServerConfig').addEventListener('click', openModal);
  $('#closeServerConfig').addEventListener('click', closeModal);
  $('#cancelServerConfig').addEventListener('click', closeModal);
  $('#importServerConfig').addEventListener('click', importSelected);
  $('#serverConfigRefresh').addEventListener('click', () => loadDirectory().catch(showError));
  $('#serverConfigUp').addEventListener('click', () => {
    const parts = currentPath.split('/').filter(Boolean);
    parts.pop();
    loadDirectory(parts.join('/')).catch(showError);
  });
  rootSelect.addEventListener('change', () => { currentPath = ''; loadDirectory('').catch(showError); });
  modal.addEventListener('click', event => { if (event.target === modal) closeModal(); });
  window.addEventListener('keydown', event => { if (event.key === 'Escape' && !modal.hidden) closeModal(); });
})();
