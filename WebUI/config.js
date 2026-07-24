(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const field = (config, path, label, value, unit = '', type = 'number', extra = {}) => ({ config, path, label, value, unit, type, ...extra });
  const sim = (path, label, value, unit = '', type = 'number', extra = {}) => field('simenv', path, label, value, unit, type, extra);
  const test = (path, label, value, unit = '', type = 'number', extra = {}) => field('test', path, label, value, unit, type, extra);

  const sections = [
    {
      title: '时间与初始状态', subtitle: 'SIMENV / TIMING', open: true,
      fields: [
        sim('seed', '仿真随机种子', 20260721, '', 'integer'),
        sim('timing.physics_hz.value', '物理更新频率', 5000, 'Hz', 'integer'),
        sim('timing.control_hz.value', '控制器频率', 500, 'Hz', 'integer'),
        sim('initial_state.position_n.value', '初始位置 NED', [0, 0, 0], 'm', 'vector'),
        sim('initial_state.velocity_n.value', '初始速度 NED', [0, 0, 0], 'm/s', 'vector'),
        sim('initial_state.attitude_q_wb.value', '初始姿态四元数', [1, 0, 0, 0], '', 'vector'),
        sim('initial_state.angular_velocity_b.value', '初始机体系角速度', [0, 0, 0], 'rad/s', 'vector')
      ]
    },
    {
      title: '机体参数', subtitle: 'SIMENV / BODY', open: true,
      fields: [
        sim('body.mass.value', '整机质量', 2.4, 'kg'),
        sim('body.center_of_mass_b.value', '质心 FRD', [0, 0, 0.08], 'm', 'vector'),
        sim('body.inertia_diagonal_b.value', '三轴转动惯量', [0.030, 0.028, 0.012], 'kg·m²', 'vector'),
        sim('body.mass.randomization.distribution', '质量随机分布', 'normal', '', 'select', { options: ['none', 'normal', 'uniform'] }),
        sim('body.mass.randomization.mode', '质量随机模式', 'relative', '', 'select', { options: ['relative', 'absolute'] }),
        sim('body.mass.randomization.mean', '质量随机均值', 0),
        sim('body.mass.randomization.stddev', '质量随机标准差', 0.05),
        sim('body.mass.randomization.clip', '质量随机截断', [-0.2, 0.2], '', 'vector')
      ]
    },
    ...['upper', 'lower'].map((name, index) => ({
      title: `${name === 'upper' ? '上' : '下'}桨电机`, subtitle: `SIMENV / MOTORS[${index}]`,
      fields: [
        sim(`motors.${index}.pwm_deadzone.value`, 'PWM 死区', 0.08),
        sim(`motors.${index}.pwm_to_rpm_table.value`, 'PWM—转速表', [[0, 0], [0.08, 0], [0.5, 900], [1, 1800]], 'rad/s', 'matrix'),
        sim(`motors.${index}.time_constant.value`, '一阶时间常数', index ? 0.050 : 0.030, 's'),
        sim(`motors.${index}.torque_coefficient.value`, '反扭矩系数', index ? 1.10e-7 : 9.8765432e-8, 'N·m/(rad/s)²'),
        sim(`motors.${index}.noise.distribution`, '转速噪声分布', 'normal', '', 'select', { options: ['normal'] }),
        sim(`motors.${index}.noise.stddev.value`, '转速噪声标准差', 5, 'rad/s')
      ]
    })),
    {
      title: '三路舵机', subtitle: 'SIMENV / SERVOS',
      fields: [0, 1, 2].flatMap(index => [
        sim(`servos.${index}.pwm_angle_table.value`, `舵机 ${index + 1} PWM—角度表`, [[-1, -0.35], [0, 0], [1, 0.35]], 'rad', 'matrix'),
        sim(`servos.${index}.tau.value`, `舵机 ${index + 1} 时间常数`, 0.020, 's'),
        sim(`servos.${index}.max_speed.value`, `舵机 ${index + 1} 最大速度`, 8.0, 'rad/s'),
        sim(`servos.${index}.backlash.value`, `舵机 ${index + 1} 回差`, 0.010, 'rad'),
        sim(`servos.${index}.deadzone.value`, `舵机 ${index + 1} 死区`, 0.015)
      ])
    },
    {
      title: '推力与气动分配', subtitle: 'SIMENV / AERODYNAMICS',
      fields: [
        sim('aerodynamics.thrust_coefficients.value', '总推力系数 [k1,k2,k3]', [4e-6, 4e-6, 2e-6], 'N/(rad/s)²', 'vector'),
        sim('aerodynamics.neutral_thrust_direction_b.value', '中立推力方向 FRD', [0, 0, -1], '', 'vector'),
        sim('aerodynamics.direct_thrust_center_b.value', '直接推力作用点 FRD', [0, 0, 0.20], 'm', 'vector'),
        sim('aerodynamics.thrust_partition.direct.value', '直接推力占比', 0.40),
        sim('aerodynamics.thrust_partition.grid_1.value', '格栅 1 推力占比', 0.20),
        sim('aerodynamics.thrust_partition.grid_2.value', '格栅 2 推力占比', 0.20),
        sim('aerodynamics.thrust_partition.grid_3.value', '格栅 3 推力占比', 0.20),
        sim('aerodynamics.coupling_attenuation.value', '格栅耦合衰减矩阵', [[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]], '', 'matrix')
      ]
    },
    {
      title: '气动格栅', subtitle: 'SIMENV / GRIDS',
      fields: [
        [[0.10, 0, 0.25], [1, 0, 0]],
        [[-0.05, 0.087, 0.25], [-0.5, 0.8660254, 0]],
        [[-0.05, -0.087, 0.25], [-0.5, -0.8660254, 0]]
      ].flatMap((values, index) => [
        sim(`aerodynamics.grids.${index}.aerodynamic_center_b.value`, `格栅 ${index + 1} 气动中心`, values[0], 'm', 'vector'),
        sim(`aerodynamics.grids.${index}.deflection_axis_b.value`, `格栅 ${index + 1} 偏转轴`, values[1], '', 'vector'),
        sim(`aerodynamics.grids.${index}.self_attenuation_curve.value`, `格栅 ${index + 1} 自衰减曲线`, [[0, 1], [0.35, 0.85]], '', 'matrix'),
        sim(`aerodynamics.grids.${index}.vector_deflection.gain.value`, `格栅 ${index + 1} 矢量增益`, 1),
        sim(`aerodynamics.grids.${index}.vector_deflection.offset.value`, `格栅 ${index + 1} 矢量偏置`, 0, 'rad')
      ])
    },
    {
      title: '传感器', subtitle: 'SIMENV / SENSORS',
      fields: [
        ['gyro', '陀螺仪', [0.002, 0.002, 0.002], [0, 0, 0], 0.001],
        ['accelerometer', '加速度计', [0.02, 0.02, 0.02], [0, 0, 0], 0.001],
        ['motor_speed', '电机转速', [2, 2], [0, 0], 0]
      ].flatMap(([key, label, noise, bias, delay]) => [
        sim(`sensors.${key}.sample_hz.value`, `${label}采样率`, 5000, 'Hz', 'integer'),
        sim(`sensors.${key}.noise.distribution`, `${label}噪声分布`, 'normal', '', 'select', { options: ['normal'] }),
        sim(`sensors.${key}.noise.stddev.value`, `${label}噪声标准差`, noise, '', 'vector'),
        sim(`sensors.${key}.bias.value`, `${label}零偏`, bias, '', 'vector'),
        sim(`sensors.${key}.delay.value`, `${label}延迟`, delay, 's')
      ])
    },
    {
      title: '日志记录', subtitle: 'SIMENV / LOGGING',
      fields: [
        sim('logging.directory', '日志目录', '../logs', '', 'text'),
        sim('logging.chunk_steps', '单日志块步数', 1024, '', 'integer'),
        sim('logging.queue_chunks', '后台队列容量', 2, '', 'integer'),
        sim('logging.overflow', '队列溢出策略', 'block', '', 'select', { options: ['block', 'error'] })
      ]
    },
    {
      title: '推理与观测', subtitle: 'TRAIN / RUNTIME', open: true,
      fields: [
        test('seed.base', '测试随机种子', 20260722, '', 'integer'),
        test('run.device', '运行设备', 'cpu', '', 'select', { options: ['cpu'] }),
        test('run.dtype', '张量类型', 'float32', '', 'select', { options: ['float32', 'float64'] }),
        test('environment.observation_source', '观测来源', 'truth', '', 'select', { options: ['truth', 'sensor'] }),
        test('runtime.checkpoint_path', '模型检查点（服务器）', 'controller/latest.pt', '', 'text'),
        test('runtime.telemetry_hz', '前端遥测频率', 30, 'Hz'),
        test('runtime.command_timeout_ms', '手柄失联超时', 250, 'ms'),
        test('runtime.cpu_threads', 'CPU 推理线程数', 1, '', 'integer'),
        test('model.type', '控制器模型', 'gru_actor_critic', '', 'select', { options: ['gru_actor_critic', 'mlp_actor_critic'] }),
        test('model.mlp_hidden_sizes', 'MLP 隐层', [256, 256, 128], '', 'vector'),
        test('model.encoder.hidden_sizes', 'Encoder 隐层', [128, 128], '', 'vector'),
        test('model.recurrent.hidden_size', 'GRU 隐状态维度', 128, '', 'integer'),
        test('model.actor_head.hidden_sizes', 'Actor Head 隐层', [128], '', 'vector')
      ]
    },
    {
      title: '手柄命令映射', subtitle: 'TRAIN / COMMAND SOURCE', open: true,
      fields: [
        test('command_source.seed', '命令源随机种子', 51002, '', 'integer'),
        test('command_source.params.throttle.minimum', '油门最小值', 0.20),
        test('command_source.params.throttle.maximum', '油门最大值', 0.85),
        test('command_source.params.throttle.spool.duration_s', '启动缓升时间', 1.0, 's'),
        test('command_source.params.throttle.spool.target_range', '启动油门范围', [0.30, 0.40], '', 'vector'),
        test('command_source.params.throttle.slew_rate.rise_per_s', '油门最大上升率', 0.50, '/s'),
        test('command_source.params.throttle.slew_rate.fall_per_s', '油门最大下降率', 0.35, '/s'),
        test('command_source.params.sticks.roll.mode', '横滚命令模式', 'angle', '', 'select', { options: ['angle', 'rate'] }),
        test('command_source.params.sticks.roll.limit_rad', '横滚角限制', 0.35, 'rad'),
        test('command_source.params.sticks.roll.time_constant_s', '横滚滤波常数', 0.20, 's'),
        test('command_source.params.sticks.pitch.mode', '俯仰命令模式', 'angle', '', 'select', { options: ['angle', 'rate'] }),
        test('command_source.params.sticks.pitch.limit_rad', '俯仰角限制', 0.35, 'rad'),
        test('command_source.params.sticks.pitch.time_constant_s', '俯仰滤波常数', 0.20, 's'),
        test('command_source.params.sticks.yaw.mode', '偏航命令模式', 'rate', '', 'select', { options: ['rate', 'angle'] }),
        test('command_source.params.sticks.yaw.limit_rad_s', '偏航角速度限制', 1.50, 'rad/s'),
        test('command_source.params.sticks.yaw.time_constant_s', '偏航滤波常数', 0.30, 's')
      ]
    },
    {
      title: '虚拟飞手高度逻辑', subtitle: 'TRAIN / HEIGHT LOGIC',
      fields: [
        test('command_source.params.throttle.height_logic.enabled', '启用高度油门逻辑', false, '', 'boolean'),
        test('command_source.params.throttle.height_logic.low_enter_m', '低高度进入阈值', 3, 'm'),
        test('command_source.params.throttle.height_logic.low_exit_m', '低高度退出阈值', 4, 'm'),
        test('command_source.params.throttle.height_logic.high_exit_m', '高高度退出阈值', 8, 'm'),
        test('command_source.params.throttle.height_logic.high_enter_m', '高高度进入阈值', 9, 'm'),
        test('command_source.params.throttle.height_logic.climb_throttle_range', '爬升油门范围', [0.58, 0.68], '', 'vector'),
        test('command_source.params.throttle.height_logic.cruise_throttle_range', '巡航油门范围', [0.48, 0.60], '', 'vector'),
        test('command_source.params.throttle.height_logic.descend_throttle_range', '下降油门范围', [0.35, 0.48], '', 'vector')
      ]
    },
    {
      title: '任务与安全终止', subtitle: 'TRAIN / TASK', open: true,
      fields: [
        test('task.episode_duration_s', '单次测试时长', 30, 's'),
        test('task.termination.max_tilt_rad', '最大倾角', 1.3, 'rad'),
        test('task.termination.max_angular_rate_rad_s', '最大角速度', 20, 'rad/s')
      ]
    },
    {
      title: '随机化', subtitle: 'TRAIN / RANDOMIZATION',
      fields: [
        test('randomization.static.seed', '静态随机种子', 31002, '', 'integer'),
        test('randomization.static.parameters.body.mass.baseline', '质量基准', 2.4, 'kg'),
        test('randomization.static.parameters.body.mass.range', '质量随机范围', [2.16, 2.64], 'kg', 'vector'),
        test('randomization.static.parameters.body.mass.distribution', '质量随机分布', 'uniform', '', 'select', { options: ['uniform', 'normal'] }),
        test('randomization.dynamic.seed', '动态随机种子', 41002, '', 'integer'),
        test('randomization.dynamic.parameters.sensors.gyro.noise.stddev.baseline', '陀螺噪声基准', [0.002, 0.002, 0.002], 'rad/s', 'vector'),
        test('randomization.dynamic.parameters.sensors.gyro.noise.stddev.stddev', '陀螺噪声随机标准差', [0.0002, 0.0002, 0.0002], 'rad/s', 'vector'),
        test('randomization.dynamic.parameters.sensors.gyro.noise.stddev.valid_range', '陀螺噪声有效范围', [0, 0.01], 'rad/s', 'vector')
      ]
    },
    {
      title: '奖励诊断', subtitle: 'TRAIN / REWARD',
      fields: [
        test('reward.calculator.params.attitude_weight', '姿态误差权重', 4.0),
        test('reward.calculator.params.angular_rate_weight', '角速度权重', 0.1),
        test('reward.calculator.params.action_rate_weight', '动作变化率权重', 0.01),
        test('reward.calculator.params.saturation_weight', '动作饱和权重', 0.02),
        test('reward.calculator.params.alive_bonus', '存活奖励', 0.1),
        test('reward.calculator.params.termination_penalty', '终止惩罚', 0.0)
      ]
    }
  ];

  function formatValue(item) {
    if (item.type === 'vector') return item.value.join(', ');
    if (item.type === 'matrix') return item.value.map(row => row.join(', ')).join(';\n');
    return String(item.value);
  }

  function renderField(item) {
    const attrs = `data-config="${item.config}" data-path="${item.path}" data-type="${item.type}" data-default="${encodeURIComponent(JSON.stringify(item.value))}"`;
    let control;
    if (item.type === 'select') {
      control = `<select ${attrs}>${item.options.map(option => `<option${option === item.value ? ' selected' : ''}>${option}</option>`).join('')}</select>`;
    } else if (item.type === 'boolean') {
      control = `<label class="config-switch"><input type="checkbox" ${attrs}${item.value ? ' checked' : ''}><i></i></label>`;
    } else if (item.type === 'matrix') {
      control = `<div class="config-matrix"><textarea rows="${Math.max(3, item.value.length)}" spellcheck="false" ${attrs}>${formatValue(item)}</textarea>${item.unit ? `<small>单位：${item.unit}</small>` : '<small>每行用分号分隔，行内元素用逗号分隔</small>'}</div>`;
    } else {
      const inputType = ['number', 'integer'].includes(item.type) ? 'number' : 'text';
      const step = item.type === 'integer' ? '1' : 'any';
      control = `<div class="unit-input config-input"><input type="${inputType}" step="${step}" value="${formatValue(item)}" ${attrs}>${item.unit ? `<b title="${item.unit}">${item.unit}</b>` : ''}</div>`;
    }
    return `<label class="config-field"><span>${item.label}<small>${item.path}</small></span>${control}</label>`;
  }

  function renderSections() {
    $('#configSections').innerHTML = sections.map(section => `
      <details${section.open ? ' open' : ''} class="config-section">
        <summary><span>${section.title}<small>${section.subtitle}</small></span><em>${section.fields.length}</em><i></i></summary>
        <div class="detail-body">${section.fields.map(renderField).join('')}</div>
      </details>`).join('');
  }

  function parseValue(input) {
    const type = input.dataset.type;
    if (type === 'boolean') return input.checked;
    if (type === 'integer') return Number.parseInt(input.value, 10);
    if (type === 'number') return Number(input.value);
    if (type === 'vector') {
      const text = input.value.trim();
      if (text.startsWith('[')) return JSON.parse(text);
      return text.split(',').map(value => value.trim()).filter(Boolean).map(Number);
    }
    if (type === 'matrix') {
      const text = input.value.trim();
      if (text.startsWith('[[')) return JSON.parse(text);
      return text.split(/;|\n/).map(row => row.trim()).filter(Boolean)
        .map(row => row.split(',').map(value => value.trim()).filter(Boolean).map(Number));
    }
    return input.value;
  }

  function setPath(root, path, value) {
    const parts = path.split('.');
    let target = root;
    parts.forEach((part, index) => {
      const last = index === parts.length - 1;
      const nextIsIndex = /^\d+$/.test(parts[index + 1] || '');
      if (last) target[part] = value;
      else {
        if (target[part] === undefined) target[part] = nextIsIndex ? [] : {};
        target = target[part];
      }
    });
  }

  function baseSimEnv() {
    return {
      schema_version: 1,
      parallel: { independent_rng: true },
      motors: [{ name: 'upper' }, { name: 'lower' }],
      servos: [{ name: 'servo_1' }, { name: 'servo_2' }, { name: 'servo_3' }],
      aerodynamics: { grids: [{ name: 'grid_1' }, { name: 'grid_2' }, { name: 'grid_3' }] }
    };
  }

  function baseTestConfig(simFilename) {
    return {
      schema_version: 2,
      experiment: { name: 'controller_webui_test', entrypoint: { type: 'flight_train.runner:run_experiment', version: 'train-framework-v1' } },
      seed: { deterministic_algorithms: true, cudnn_benchmark: false },
      run: { allow_unimplemented_simulator: false },
      environment: { factory: 'simenv:SimulationEnvironment', config_path: simFilename },
      command_source: { type: 'flight_train.commands:VirtualPilotCommandSource', version: '1', params: { sticks: { target_sampling: { distribution: 'centered', center_exponent: 2, hold_duration_s: { range: [1, 4] } }, reset: { filtered_stick: 'zero', initial_target_scale: 0.25 } } } },
      control_contract: {
        version: 'self_stabilize_v1', observation_profile: 'attitude_self_stabilize_21d_v2',
        policy_action: { fields: ['lower_motor', 'servo_1', 'servo_2', 'servo_3'] },
        external_action: { fields: ['upper_motor'] },
        simulator_command: { fields: ['upper_motor', 'lower_motor', 'servo_1', 'servo_2', 'servo_3'] }
      },
      randomization: {
        static: { scope: 'episode', parameters: { body: { mass: { unit: 'kg', seed_stream: 'static.body.mass' } } } },
        dynamic: { scope: 'simulator', parameters: { sensors: { gyro: { noise: { stddev: { distribution: 'normal', on_out_of_range: 'clamp', unit: 'rad/s', seed_stream: 'dynamic.sensor.gyro.noise' } } } } } }
      },
      reward: {
        calculator: { type: 'flight_train.rewards.attitude:AttitudeRewardCalculator', version: '1', params: {} },
        context: { fields: [{ name: 'attitude_geodesic_rad', source: 'task' }, { name: 'angular_velocity_b', source: 'sensor' }, { name: 'static_domain', source: 'train_randomizer', visibility: 'reward_only' }] }
      },
      evaluation: {}, checkpoint: {}, recording: {}
    };
  }

  function validate(simConfig, testConfig) {
    const physicsHz = simConfig.timing.physics_hz.value;
    const controlHz = simConfig.timing.control_hz.value;
    if (!Number.isInteger(physicsHz) || !Number.isInteger(controlHz) || physicsHz <= 0 || controlHz <= 0 || physicsHz % controlHz !== 0) {
      throw new Error('physics_hz 和 control_hz 必须为正整数，且 control_hz 必须整除 physics_hz。');
    }
    const partition = simConfig.aerodynamics.thrust_partition;
    const sum = ['direct', 'grid_1', 'grid_2', 'grid_3'].reduce((total, key) => total + partition[key].value, 0);
    if (Math.abs(sum - 1) > 1e-6) throw new Error('direct 与三个 grid 的推力占比之和必须等于 1。');
    if (testConfig.task.episode_duration_s <= 0) throw new Error('测试时长必须大于 0。');
    document.querySelectorAll('[data-config]').forEach(input => {
      const value = parseValue(input);
      const values = Array.isArray(value) ? value.flat(Infinity) : [value];
      if (values.some(item => typeof item === 'number' && !Number.isFinite(item))) throw new Error(`${input.dataset.path} 包含无效数值。`);
    });
  }

  function scalar(value) {
    if (typeof value === 'string') {
      if (/^[A-Za-z0-9_./:+-]+$/.test(value) && !['true', 'false', 'null'].includes(value)) return value;
      return JSON.stringify(value);
    }
    if (Array.isArray(value)) return JSON.stringify(value);
    if (value === null) return 'null';
    return String(value);
  }

  function toYaml(value, indent = 0) {
    const pad = ' '.repeat(indent);
    if (Array.isArray(value)) {
      if (!value.length) return `${pad}[]`;
      return value.map(item => typeof item === 'object' && item !== null
        ? `${pad}-\n${toYaml(item, indent + 2)}`
        : `${pad}- ${scalar(item)}`).join('\n');
    }
    return Object.entries(value).map(([key, item]) => {
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        const entries = Object.keys(item);
        return entries.length ? `${pad}${key}:\n${toYaml(item, indent + 2)}` : `${pad}${key}: {}`;
      }
      return `${pad}${key}: ${scalar(item)}`;
    }).join('\n');
  }

  function buildBundle() {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
    const simFilename = `simenv_webui_${stamp}.yaml`;
    const testFilename = `controller_test_${stamp}.yaml`;
    const simConfig = baseSimEnv();
    const testConfig = baseTestConfig(simFilename);
    document.querySelectorAll('[data-config]').forEach(input => {
      setPath(input.dataset.config === 'simenv' ? simConfig : testConfig, input.dataset.path, parseValue(input));
    });
    const staticMass = testConfig.randomization.static.parameters.body.mass;
    testConfig.randomization.static.parameters = { 'body.mass': staticMass };
    const dynamicGyro = testConfig.randomization.dynamic.parameters.sensors.gyro.noise.stddev;
    testConfig.randomization.dynamic.parameters = { 'sensors.gyro.noise.stddev': dynamicGyro };
    if (testConfig.model.type === 'mlp_actor_critic') {
      testConfig.model = { type: 'mlp_actor_critic', hidden_sizes: testConfig.model.mlp_hidden_sizes };
    } else {
      delete testConfig.model.mlp_hidden_sizes;
    }
    validate(simConfig, testConfig);
    const simYaml = `# Generated by RL Flight WebUI\n${toYaml(simConfig)}\n`;
    const testYaml = `# Generated by RL Flight WebUI\n${toYaml(testConfig)}\n`;
    return {
      generatedAt: new Date().toISOString(),
      simenv: { filename: simFilename, config: simConfig, yaml: simYaml },
      test: { filename: testFilename, config: testConfig, yaml: testYaml }
    };
  }

  let latestBundle = null;
  function prepareSimulationStart() {
    latestBundle = buildBundle();
    const indicator = $('#dirtyIndicator');
    indicator.classList.remove('dirty');
    indicator.querySelector('span').textContent = `已生成 ${latestBundle.simenv.filename}`;
    window.dispatchEvent(new CustomEvent('rlflightconfigurationready', { detail: latestBundle }));
    return latestBundle;
  }

  function download(filename, content) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([content], { type: 'application/yaml;charset=utf-8' }));
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
  }

  function restoreDefaults() {
    document.querySelectorAll('[data-default]').forEach(input => {
      const value = JSON.parse(decodeURIComponent(input.dataset.default));
      if (input.type === 'checkbox') input.checked = value;
      else if (Array.isArray(value)) input.value = Array.isArray(value[0]) ? value.map(row => row.join(', ')).join(';\n') : value.join(', ');
      else input.value = String(value);
    });
  }

  function getPath(root, path) {
    return path.split('.').reduce((value, part) => value === undefined || value === null ? undefined : value[part], root);
  }

  function normalizeImportedConfig(kind, source) {
    const config = typeof structuredClone === 'function' ? structuredClone(source) : JSON.parse(JSON.stringify(source));
    if (kind !== 'test') return config;
    const staticParameters = config.randomization?.static?.parameters;
    if (staticParameters?.['body.mass']) {
      staticParameters.body = { mass: staticParameters['body.mass'] };
    }
    const dynamicParameters = config.randomization?.dynamic?.parameters;
    if (dynamicParameters?.['sensors.gyro.noise.stddev']) {
      dynamicParameters.sensors = { gyro: { noise: { stddev: dynamicParameters['sensors.gyro.noise.stddev'] } } };
    }
    if (config.model?.type === 'mlp_actor_critic' && config.model.hidden_sizes) {
      config.model.mlp_hidden_sizes = config.model.hidden_sizes;
    }
    return config;
  }

  function setInputValue(input, value) {
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else if (Array.isArray(value)) {
      input.value = Array.isArray(value[0]) ? value.map(row => row.join(', ')).join(';\n') : value.join(', ');
    } else if (input.tagName === 'SELECT') {
      if ([...input.options].some(option => option.value === String(value))) input.value = String(value);
      else return false;
    } else input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  }

  function inputDefault(input) {
    return JSON.parse(decodeURIComponent(input.dataset.default));
  }

  function hasImportedValue(value) {
    if (value === undefined || value === null) return false;
    if (typeof value === 'string' && !value.trim()) return false;
    if (Array.isArray(value) && value.length === 0) return false;
    return true;
  }

  function importConfig(kind, source, sourceLabel = '服务器配置') {
    if (!['simenv', 'test'].includes(kind)) throw new Error('无法判断配置类型，请选择 SimEnv 或 Train/controller-test 配置。');
    const config = normalizeImportedConfig(kind, source);
    let applied = 0;
    let defaulted = 0;
    document.querySelectorAll(`[data-config="${kind}"]`).forEach(input => {
      const value = getPath(config, input.dataset.path);
      if (hasImportedValue(value) && setInputValue(input, value)) {
        applied += 1;
      } else {
        setInputValue(input, inputDefault(input));
        defaulted += 1;
      }
    });
    if (!applied) throw new Error('文件中没有与当前参数面板匹配的字段。');
    latestBundle = null;
    const indicator = $('#dirtyIndicator');
    indicator.classList.add('dirty');
    indicator.querySelector('span').textContent = `已导入 ${sourceLabel} · ${applied} 项，默认补齐 ${defaulted} 项`;
    window.dispatchEvent(new CustomEvent('rlflightconfigimported', { detail: { kind, source: sourceLabel, applied, defaulted, config: source } }));
    return { kind, applied, defaulted, total: applied + defaulted };
  }

  renderSections();
  $('#exportConfig').addEventListener('click', () => {
    try {
      const bundle = prepareSimulationStart();
      download(bundle.simenv.filename, bundle.simenv.yaml);
      download(bundle.test.filename, bundle.test.yaml);
    } catch (error) {
      $('#dirtyIndicator').classList.add('dirty');
      $('#dirtyIndicator span').textContent = error.message;
    }
  });

  window.RLFlightConfig = { buildBundle, prepareSimulationStart, restoreDefaults, importConfig, getLatestBundle: () => latestBundle };
})();
