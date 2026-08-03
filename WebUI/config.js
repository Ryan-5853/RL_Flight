(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const field = (config, path, label, value, unit = '', type = 'number', extra = {}) => ({ config, path, label, value, unit, type, ...extra });
  const sim = (path, label, value, unit = '', type = 'number', extra = {}) => field('simenv', path, label, value, unit, type, extra);
  const test = (path, label, value, unit = '', type = 'number', extra = {}) => field('test', path, label, value, unit, type, extra);
  const defaultMotorTable = [[0, 0], [0.08, 0], [0.5, 900], [1, 1800]];
  const defaultServoTable = [[-1, -0.35], [0, 0], [1, 0.35]];
  const defaultGridGeometry = [
    [[0.10, 0, 0.25], [1, 0, 0]],
    [[-0.05, 0.087, 0.25], [-0.5, 0.8660254, 0]],
    [[-0.05, -0.087, 0.25], [-0.5, -0.8660254, 0]]
  ];

  const sections = [
    {
      title: '时间与初始状态', subtitle: 'SIMENV / TIMING', open: true,
      fields: [
        sim('seed', '仿真随机种子', 20260721, '', 'integer'),
        sim('timing.physics_hz.value', '物理更新频率', 500, 'Hz', 'integer'),
        sim('timing.control_hz.value', '控制器频率', 500, 'Hz', 'integer'),
        sim('initial_state.position_n.value', '初始位置 NED', [0, 0, 0], 'm', 'vector'),
        sim('initial_state.velocity_n.value', '初始速度 NED', [0, 0, 0], 'm/s', 'vector'),
        sim('initial_state.attitude_q_wb.value', '初始姿态四元数', [0.99619693, 0.06966088, -0.05220847, 0.00365077], '', 'vector'),
        sim('initial_state.angular_velocity_b.value', '初始机体系角速度', [0.05, -0.04, 0], 'rad/s', 'vector')
      ]
    },
    {
      title: '机体参数', subtitle: 'SIMENV / BODY', open: true,
      fields: [
        sim('body.mass.value', '整机质量', 2.4, 'kg'),
        sim('body.center_of_mass_b.value', '质心 FRD [x前 y右 z下+]', [0, 0, 0.08], 'm', 'vector'),
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
        sim(`motors.${index}.pwm_to_rpm_table.value`, 'PWM—转速表', defaultMotorTable, 'rad/s', 'matrix'),
        sim(`motors.${index}.time_constant.value`, '一阶时间常数', index ? 0.050 : 0.030, 's'),
        sim(`motors.${index}.torque_coefficient.value`, '反扭矩系数', index ? 1.10e-7 : 9.8765432e-8, 'N·m/(rad/s)²'),
        sim(`motors.${index}.noise.distribution`, '转速噪声分布', 'normal', '', 'select', { options: ['normal'] }),
        sim(`motors.${index}.noise.stddev.value`, '转速噪声标准差', 5, 'rad/s')
      ]
    })),
    {
      title: '三路舵机', subtitle: 'SIMENV / SERVOS',
      fields: [0, 1, 2].flatMap(index => [
        sim(`servos.${index}.pwm_angle_table.value`, `舵机 ${index + 1} PWM—角度表`, defaultServoTable, 'rad', 'matrix'),
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
        ...defaultGridGeometry
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
        sim(`sensors.${key}.sample_hz.value`, `${label}采样率`, 500, 'Hz', 'integer'),
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
        test('runtime.checkpoint_path', '推理包路径（服务器）', '', '', 'select', { options: [''] }),
        test('runtime.compile_kernels', '编译实时仿真内核', true, '', 'boolean'),
        test('runtime.warmup_steps', '实时内核预热步数', 3, '', 'integer'),
        test('runtime.spin_us', '截止时间自旋窗口', 200, 'μs'),
        test('runtime.execution_hz', '墙钟执行频率', 500, 'Hz'),
        test('runtime.telemetry_hz', '前端遥测频率', 60, 'Hz'),
        test('runtime.command_timeout_ms', '手柄失联安全切换', 5000, 'ms'),
        test('runtime.cpu_threads', 'CPU 推理线程数', 1, '', 'integer'),
        test('model.type', '控制器模型', 'gru_actor_critic', '', 'select', { options: ['gru_actor_critic', 'mlp_actor_critic'] }),
        test('model.mlp_hidden_sizes', 'MLP 隐层', [256, 256, 128], '', 'vector'),
        test('model.encoder.hidden_sizes', 'Encoder 隐层', [128, 128], '', 'vector'),
        test('model.recurrent.hidden_size', 'GRU 隐状态维度', 128, '', 'integer'),
        test('model.actor_head.hidden_sizes', 'Actor Head 隐层', [128], '', 'vector')
      ]
    },
    {
      title: '统一控制器', subtitle: 'CONTROLLER / CONTROL LAW', open: true,
      fields: [
        test('controller.type', '控制器类型', 'hybrid_pid_lqr', '', 'select', { options: ['hybrid_pid_lqr', 'pid', 'lqr', 'neural'] }),
        test('controller.params.flight_mode', '飞行参考模式', 'attitude', '', 'select', { options: ['attitude', 'position'] }),
        test('controller.params.collective_mode', '总推力模式', 'hover', '', 'select', { options: ['hover', 'manual'] }),
        test('controller.params.position.kp', '位置外环 · Kp', 1.0, 's⁻²'),
        test('controller.params.position.kd', '位置外环 · Kd', 1.6, 's⁻¹'),
        test('controller.params.position.maximum_acceleration_m_s2', '位置外环 · 最大水平加速度', 3.0, 'm/s²'),
        test('controller.params.position.maximum_tilt_rad', '位置外环 · 最大倾角', 0.35, 'rad'),
        test('controller.params.pid.altitude.kp', '实时 Hover 高度 PID · Kp', 4.0),
        test('controller.params.pid.altitude.ki', '实时 Hover 高度 PID · Ki', 0.8),
        test('controller.params.pid.altitude.kd', '实时 Hover 高度 PID · Kd', 3.6),
        test('controller.params.pid.attitude.roll_pitch_natural_frequency_rad_s', '横滚/俯仰自然频率', 6.0, 'rad/s'),
        test('controller.params.pid.attitude.yaw_rate_bandwidth_rad_s', '偏航角速度带宽', 5.0, 'rad/s'),
        test('controller.params.pid.attitude.damping_ratio', '姿态阻尼比', 0.85),
        test('controller.params.lqr.input_weight_scale', 'LQR 输入代价倍率', 0.25),
        test('controller.params.hybrid.lqr_enter_motor_speed_fraction', 'LQR 接管转速比例', 0.78),
        test('controller.params.hybrid.lqr_enter_attitude_error_rad', 'LQR 接管姿态误差', 0.20944, 'rad'),
        test('controller.params.hybrid.transition_s', 'PID/LQR 混合时间', 0.25, 's')
      ]
    },
    {
      title: '控制器辨识模型', subtitle: 'CONTROLLER / IDENTIFIED PLANT', syncAction: true,
      fields: [
        test('controller.params.model_parameters.source', '参数来源', 'manual', '', 'select', { options: ['manual', 'synchronized'] }),
        test('controller.params.model_parameters.manual.body.mass', '辨识质量', 2.4, 'kg'),
        test('controller.params.model_parameters.manual.body.center_of_mass_b', '辨识质心 FRD', [0, 0, 0.08], 'm', 'vector'),
        test('controller.params.model_parameters.manual.body.inertia_diagonal_b', '辨识三轴惯量', [0.030, 0.028, 0.012], 'kg·m²', 'vector'),
        test('controller.params.model_parameters.manual.motors.time_constant', '辨识电机时间常数 [上,下]', [0.030, 0.050], 's', 'vector'),
        test('controller.params.model_parameters.manual.motors.torque_coefficient', '辨识电机反扭矩系数 [上,下]', [9.8765432e-8, 1.10e-7], 'N·m/(rad/s)²', 'vector'),
        test('controller.params.model_parameters.manual.motors.upper_pwm_to_rpm_table', '辨识上桨 PWM—转速表', defaultMotorTable, 'rad/s', 'matrix'),
        test('controller.params.model_parameters.manual.motors.lower_pwm_to_rpm_table', '辨识下桨 PWM—转速表', defaultMotorTable, 'rad/s', 'matrix'),
        test('controller.params.model_parameters.manual.servos.tau', '辨识舵机时间常数', [0.020, 0.020, 0.020], 's', 'vector'),
        ...[1, 2, 3].map(index => test(`controller.params.model_parameters.manual.servos.servo_${index}_pwm_angle_table`, `辨识舵机 ${index} PWM—角度表`, defaultServoTable, 'rad', 'matrix')),
        test('controller.params.model_parameters.manual.aerodynamics.thrust_coefficients', '辨识总推力系数 [k1,k2,k3]', [4e-6, 4e-6, 2e-6], 'N/(rad/s)²', 'vector'),
        test('controller.params.model_parameters.manual.aerodynamics.neutral_thrust_direction_b', '辨识中立推力方向 FRD', [0, 0, -1], '', 'vector'),
        test('controller.params.model_parameters.manual.aerodynamics.direct_thrust_center_b', '辨识直接推力作用点 FRD', [0, 0, 0.20], 'm', 'vector'),
        test('controller.params.model_parameters.manual.aerodynamics.thrust_partition', '辨识推力占比 [直接,格栅1,2,3]', [0.40, 0.20, 0.20, 0.20], '', 'vector'),
        test('controller.params.model_parameters.manual.aerodynamics.coupling_attenuation', '辨识格栅耦合衰减矩阵', [[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]], '', 'matrix'),
        test('controller.params.model_parameters.manual.aerodynamics.grids.aerodynamic_center_b', '辨识三格栅气动中心', defaultGridGeometry.map(item => item[0]), 'm', 'matrix'),
        test('controller.params.model_parameters.manual.aerodynamics.grids.deflection_axis_b', '辨识三格栅偏转轴', defaultGridGeometry.map(item => item[1]), '', 'matrix'),
        ...[1, 2, 3].map(index => test(`controller.params.model_parameters.manual.aerodynamics.grids.grid_${index}_self_attenuation_curve`, `辨识格栅 ${index} 自衰减曲线`, [[0, 1], [0.35, 0.85]], '', 'matrix')),
        test('controller.params.model_parameters.manual.aerodynamics.grids.vector_deflection_gain', '辨识格栅矢量增益', [1, 1, 1], '', 'vector'),
        test('controller.params.model_parameters.manual.aerodynamics.grids.vector_deflection_offset', '辨识格栅矢量偏置', [0, 0, 0], 'rad', 'vector')
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
        test('command_source.params.sticks.roll.mode', '横滚命令模式', 'angle', '', 'select', { options: ['angle'] }),
        test('command_source.params.sticks.roll.limit_rad', '横滚角限制', 0.35, 'rad'),
        test('command_source.params.sticks.roll.time_constant_s', '横滚滤波常数', 0.20, 's'),
        test('command_source.params.sticks.pitch.mode', '俯仰命令模式', 'angle', '', 'select', { options: ['angle'] }),
        test('command_source.params.sticks.pitch.limit_rad', '俯仰角限制', 0.35, 'rad'),
        test('command_source.params.sticks.pitch.time_constant_s', '俯仰滤波常数', 0.20, 's'),
        test('command_source.params.sticks.yaw.mode', '偏航命令模式', 'rate', '', 'select', { options: ['rate'] }),
        test('command_source.params.sticks.yaw.limit_rad_s', '偏航角速度限制', 1.50, 'rad/s'),
        test('command_source.params.sticks.yaw.time_constant_s', '偏航滤波常数', 0.30, 's')
      ]
    },
    {
      title: '离线 Rollout 高度控制', subtitle: 'TRAIN / OFFLINE HEIGHT PI',
      fields: [
        test('command_source.params.throttle.height_controller.observation_source', '高度观测来源', 'truth', '', 'select', { options: ['truth'] }),
        test('command_source.params.throttle.height_controller.target_m', '目标高度', 0, 'm'),
        test('command_source.params.throttle.height_controller.initial_throttle_range', '初始油门范围', [0.48, 0.60], '', 'vector'),
        test('command_source.params.throttle.height_controller.proportional_gain', '增量 PI · Kp', 0.08),
        test('command_source.params.throttle.height_controller.integral_gain', '增量 PI · Ki', 0.04),
        test('command_source.params.throttle.height_controller.error_limit_m', '高度误差限幅', 5, 'm')
      ]
    },
    {
      title: '虚拟飞手目标采样', subtitle: 'TRAIN / TARGET SAMPLING',
      fields: [
        test('command_source.params.sticks.target_sampling.distribution', '目标采样分布', 'centered', '', 'select', { options: ['centered'] }),
        test('command_source.params.sticks.target_sampling.center_exponent', '中心分布指数', 2.0),
        test('command_source.params.sticks.target_sampling.hold_duration_s.range', '目标保持时间范围', [1.0, 4.0], 's', 'vector'),
        test('command_source.params.sticks.reset.filtered_stick', '重置后滤波杆量', 'zero', '', 'select', { options: ['zero'] }),
        test('command_source.params.sticks.reset.initial_target_scale', '初始目标幅度比例', 0.25)
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

  function fieldGroup(item) {
    if (item.config === 'simenv') return 'simenv';
    if (item.path.startsWith('controller.') || item.path.startsWith('command_source.')) return 'control';
    return 'runtime';
  }

  function renderField(item) {
    const attrs = `data-config="${item.config}" data-path="${item.path}" data-type="${item.type}" data-default="${encodeURIComponent(JSON.stringify(item.value))}"`;
    const group = fieldGroup(item);
    const searchText = `${item.label} ${item.path} ${item.unit} ${group}`.toLocaleLowerCase('zh-CN');
    let control;
    if (item.type === 'select') {
      control = `<select ${attrs}>${item.options.map(option => `<option${option === item.value ? ' selected' : ''}>${option}</option>`).join('')}</select>`;
    } else if (item.type === 'boolean') {
      control = `<span class="config-switch"><input type="checkbox" ${attrs}${item.value ? ' checked' : ''}><i></i></span>`;
    } else if (item.type === 'matrix') {
      control = `<div class="config-matrix"><textarea rows="${Math.max(3, item.value.length)}" spellcheck="false" ${attrs}>${formatValue(item)}</textarea>${item.unit ? `<small>单位：${item.unit}</small>` : '<small>每行用分号分隔，行内元素用逗号分隔</small>'}</div>`;
    } else {
      const inputType = ['number', 'integer'].includes(item.type) ? 'number' : 'text';
      const step = item.type === 'integer' ? '1' : 'any';
      control = `<div class="unit-input config-input${item.unit ? ' has-unit' : ''}"><input type="${inputType}" step="${step}" value="${formatValue(item)}" ${attrs}>${item.unit ? `<b title="${item.unit}">${item.unit}</b>` : ''}</div>`;
    }
    return `<label class="config-field" data-config-group="${group}" data-search="${encodeURIComponent(searchText)}"><span>${item.label}<small title="${item.path}">${item.path}</small></span>${control}</label>`;
  }

  function renderSections() {
    let savedState = {};
    try {
      savedState = JSON.parse(localStorage.getItem('rlflight.config.sections.v1') || '{}');
    } catch (_) {
      savedState = {};
    }
    $('#configSections').innerHTML = sections.map(section => `
      <details${savedState[section.subtitle] ?? section.open ? ' open' : ''} class="config-section" data-section-key="${encodeURIComponent(section.subtitle)}">
        <summary><span>${section.title}<small>${section.subtitle}</small></span><em data-section-count>${section.fields.length}</em><i></i></summary>
        <div class="detail-body">
          ${section.syncAction ? '<div class="config-model-sync"><button type="button" id="syncControllerModel">从 SimEnv 复制到辨识模型</button><span>复制后自动切换为 manual，可继续制造参数失配</span></div>' : ''}
          ${section.fields.map(renderField).join('')}
        </div>
      </details>`).join('');
  }

  function parseValue(input) {
    const type = input.dataset.type;
    if (type === 'boolean') return input.checked;
    if (type === 'integer') return Number.parseInt(input.value, 10);
    if (type === 'number') return input.value.trim() ? Number(input.value) : Number.NaN;
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

  let activeFilter = 'all';
  let configDirty = false;

  function valueIsFinite(value) {
    if (Array.isArray(value)) return value.length > 0 && value.every(valueIsFinite);
    return typeof value !== 'number' || Number.isFinite(value);
  }

  function inputIsValid(input) {
    try {
      const value = parseValue(input);
      if (!valueIsFinite(value)) return false;
      if (input.dataset.type === 'matrix') {
        if (!Array.isArray(value) || !value.length || value.some(row => !Array.isArray(row) || !row.length)) return false;
        return value.every(row => row.length === value[0].length);
      }
      return true;
    } catch (_) {
      return false;
    }
  }

  function inputIsModified(input) {
    try {
      return JSON.stringify(parseValue(input)) !== JSON.stringify(inputDefault(input));
    } catch (_) {
      return true;
    }
  }

  function applyParameterFilter() {
    const query = ($('#configSearch')?.value || '').trim().toLocaleLowerCase('zh-CN');
    const fields = [...document.querySelectorAll('.config-field')];
    let visible = 0;
    fields.forEach(fieldNode => {
      const searchText = decodeURIComponent(fieldNode.dataset.search || '');
      const matchesText = !query || searchText.includes(query);
      const matchesGroup = activeFilter === 'all'
        || (activeFilter === 'modified' ? fieldNode.classList.contains('modified') : fieldNode.dataset.configGroup === activeFilter);
      fieldNode.hidden = !(matchesText && matchesGroup);
      if (!fieldNode.hidden) visible += 1;
    });
    document.querySelectorAll('.config-section').forEach(section => {
      const visibleFields = [...section.querySelectorAll('.config-field:not([hidden])')];
      section.hidden = visibleFields.length === 0;
      const counter = section.querySelector('[data-section-count]');
      if (counter) counter.textContent = visibleFields.length === section.querySelectorAll('.config-field').length
        ? String(visibleFields.length)
        : `${visibleFields.length}/${section.querySelectorAll('.config-field').length}`;
      if (query && visibleFields.length) section.open = true;
    });
    const deviceSettings = $('.controller-settings');
    if (deviceSettings) deviceSettings.hidden = Boolean(query) || activeFilter !== 'all';
    const clearButton = $('#clearConfigSearch');
    if (clearButton) clearButton.hidden = !query;
    const count = $('#configVisibleCount');
    if (count) count.textContent = visible === fields.length ? `共 ${fields.length} 项参数` : `显示 ${visible} / ${fields.length} 项`;
  }

  function refreshParameterState({ markDirty = false, statusText = '' } = {}) {
    if (markDirty) {
      configDirty = true;
      latestBundle = null;
    }
    let modifiedCount = 0;
    let invalidCount = 0;
    document.querySelectorAll('[data-config]').forEach(input => {
      const fieldNode = input.closest('.config-field');
      const valid = inputIsValid(input);
      const modified = inputIsModified(input);
      fieldNode?.classList.toggle('invalid', !valid);
      fieldNode?.classList.toggle('modified', modified);
      input.setAttribute('aria-invalid', String(!valid));
      if (!valid) invalidCount += 1;
      if (modified) modifiedCount += 1;
    });
    applyParameterFilter();
    const indicator = $('#dirtyIndicator');
    const applyButton = $('#applyButton');
    indicator?.classList.toggle('invalid', invalidCount > 0);
    indicator?.classList.toggle('dirty', invalidCount === 0 && configDirty);
    $('#effectiveConfig')?.classList.toggle('stale', configDirty);
    if (applyButton) applyButton.disabled = invalidCount > 0;
    if (indicator) {
      const label = indicator.querySelector('span');
      if (invalidCount) label.textContent = `${invalidCount} 项参数无效，请先修正`;
      else if (statusText) label.textContent = statusText;
      else if (configDirty) label.textContent = modifiedCount
        ? `${modifiedCount} 项偏离默认值，尚未应用`
        : '已恢复默认参数，尚未应用';
      else if (!latestBundle) label.textContent = '参数已就绪，可直接启动仿真';
    }
    return { modifiedCount, invalidCount };
  }

  function persistSectionState() {
    const state = {};
    document.querySelectorAll('.config-section').forEach(section => {
      state[decodeURIComponent(section.dataset.sectionKey)] = section.open;
    });
    try {
      localStorage.setItem('rlflight.config.sections.v1', JSON.stringify(state));
    } catch (_) {
      // Storage can be unavailable in hardened browser contexts; UI still works.
    }
  }

  function bindParameterTools() {
    document.querySelectorAll('[data-config]').forEach(input => {
      input.addEventListener('input', () => refreshParameterState({ markDirty: true }));
      input.addEventListener('change', () => refreshParameterState({ markDirty: true }));
    });
    $('#configSearch')?.addEventListener('input', applyParameterFilter);
    $('#clearConfigSearch')?.addEventListener('click', () => {
      $('#configSearch').value = '';
      applyParameterFilter();
      $('#configSearch').focus();
    });
    document.querySelectorAll('[data-config-filter]').forEach(button => {
      button.addEventListener('click', () => {
        activeFilter = button.dataset.configFilter;
        document.querySelectorAll('[data-config-filter]').forEach(candidate => {
          const active = candidate === button;
          candidate.classList.toggle('active', active);
          candidate.setAttribute('aria-pressed', String(active));
        });
        applyParameterFilter();
      });
    });
    $('#expandConfigSections')?.addEventListener('click', () => {
      document.querySelectorAll('.config-section:not([hidden])').forEach(section => { section.open = true; });
      persistSectionState();
    });
    $('#collapseConfigSections')?.addEventListener('click', () => {
      document.querySelectorAll('.config-section').forEach(section => { section.open = false; });
      persistSectionState();
    });
    document.querySelectorAll('.config-section').forEach(section => {
      section.addEventListener('toggle', () => {
        if (!$('#configSearch')?.value.trim()) persistSectionState();
      });
    });
    $('#syncControllerModel')?.addEventListener('click', synchronizeControllerModel);
  }

  function synchronizeControllerModel() {
    const read = path => parseValue(document.querySelector(`[data-config="simenv"][data-path="${path}"]`));
    const write = (path, value) => {
      const input = document.querySelector(`[data-config="test"][data-path="controller.params.model_parameters.manual.${path}"]`);
      if (!input || !setInputValue(input, value)) throw new Error(`无法同步控制器参数 ${path}`);
    };
    write('body.mass', read('body.mass.value'));
    write('body.center_of_mass_b', read('body.center_of_mass_b.value'));
    write('body.inertia_diagonal_b', read('body.inertia_diagonal_b.value'));
    write('motors.time_constant', [0, 1].map(index => read(`motors.${index}.time_constant.value`)));
    write('motors.torque_coefficient', [0, 1].map(index => read(`motors.${index}.torque_coefficient.value`)));
    write('motors.upper_pwm_to_rpm_table', read('motors.0.pwm_to_rpm_table.value'));
    write('motors.lower_pwm_to_rpm_table', read('motors.1.pwm_to_rpm_table.value'));
    write('servos.tau', [0, 1, 2].map(index => read(`servos.${index}.tau.value`)));
    [0, 1, 2].forEach(index => write(`servos.servo_${index + 1}_pwm_angle_table`, read(`servos.${index}.pwm_angle_table.value`)));
    write('aerodynamics.thrust_coefficients', read('aerodynamics.thrust_coefficients.value'));
    write('aerodynamics.neutral_thrust_direction_b', read('aerodynamics.neutral_thrust_direction_b.value'));
    write('aerodynamics.direct_thrust_center_b', read('aerodynamics.direct_thrust_center_b.value'));
    write('aerodynamics.thrust_partition', ['direct', 'grid_1', 'grid_2', 'grid_3'].map(name => read(`aerodynamics.thrust_partition.${name}.value`)));
    write('aerodynamics.coupling_attenuation', read('aerodynamics.coupling_attenuation.value'));
    write('aerodynamics.grids.aerodynamic_center_b', [0, 1, 2].map(index => read(`aerodynamics.grids.${index}.aerodynamic_center_b.value`)));
    write('aerodynamics.grids.deflection_axis_b', [0, 1, 2].map(index => read(`aerodynamics.grids.${index}.deflection_axis_b.value`)));
    [0, 1, 2].forEach(index => write(`aerodynamics.grids.grid_${index + 1}_self_attenuation_curve`, read(`aerodynamics.grids.${index}.self_attenuation_curve.value`)));
    write('aerodynamics.grids.vector_deflection_gain', [0, 1, 2].map(index => read(`aerodynamics.grids.${index}.vector_deflection.gain.value`)));
    write('aerodynamics.grids.vector_deflection_offset', [0, 1, 2].map(index => read(`aerodynamics.grids.${index}.vector_deflection.offset.value`)));
    const source = document.querySelector('[data-config="test"][data-path="controller.params.model_parameters.source"]');
    setInputValue(source, 'manual');
    refreshParameterState({ markDirty: true, statusText: '已复制 SimEnv 参数到手动辨识模型，可继续修改失配项' });
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
      aerodynamics: { grids: [{ name: 'grid_1' }, { name: 'grid_2' }, { name: 'grid_3' }] },
      // At 500 Hz the default 1 ms gyro/accelerometer delay is half a
      // physics step. SimEnv requires the interpolation policy to be explicit
      // for every fractional-step delay.
      sensors: {
        gyro: { interpolation: 'linear' },
        accelerometer: { interpolation: 'linear' },
        motor_speed: { interpolation: 'linear' }
      }
    };
  }

  function baseTestConfig(simFilename) {
    return {
      schema_version: 2,
      experiment: { name: 'controller_webui_test', entrypoint: { type: 'flight_train.runner:run_experiment', version: 'train-framework-v1' } },
      seed: { deterministic_algorithms: true, cudnn_benchmark: false },
      run: { allow_unimplemented_simulator: false },
      environment: { factory: 'simenv:SimulationEnvironment', config_path: simFilename },
      controller: { type: 'hybrid_pid_lqr', params: {} },
      command_source: { type: 'flight_train.commands:VirtualPilotCommandSource', version: '2', params: { sticks: { target_sampling: { distribution: 'centered', center_exponent: 2, hold_duration_s: { range: [1, 4] } }, reset: { filtered_stick: 'zero', initial_target_scale: 0.25 } } } },
      control_contract: {
        version: 'self_stabilize_v1', observation_profile: 'attitude_self_stabilize_21d_v3',
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
    if (physicsHz !== 500 || controlHz !== 500) {
      throw new Error('当前 SimEnv 为单步 500 Hz 接口，physics_hz 和 control_hz 必须同时等于 500。');
    }
    const partition = simConfig.aerodynamics.thrust_partition;
    const sum = ['direct', 'grid_1', 'grid_2', 'grid_3'].reduce((total, key) => total + partition[key].value, 0);
    if (Math.abs(sum - 1) > 1e-6) throw new Error('direct 与三个 grid 的推力占比之和必须等于 1。');
    if (testConfig.task.episode_duration_s <= 0) throw new Error('测试时长必须大于 0。');
    if (testConfig.controller.params.flight_mode === 'position' && testConfig.controller.params.collective_mode !== 'hover') {
      throw new Error('position 位置模式要求总推力模式为 hover。');
    }
    if (testConfig.controller.type === 'neural' && !String(testConfig.runtime.checkpoint_path || '').trim()) {
      throw new Error('神经网络控制器必须选择服务器部署推理包。');
    }
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
    configDirty = false;
    refreshParameterState();
    const indicator = $('#dirtyIndicator');
    indicator.classList.remove('dirty', 'invalid');
    indicator.querySelector('span').textContent = `配置已应用 · ${latestBundle.simenv.filename}`;
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
    refreshParameterState({ markDirty: true, statusText: '已恢复默认参数，等待应用' });
  }

  function getPath(root, path) {
    return path.split('.').reduce((value, part) => value === undefined || value === null ? undefined : value[part], root);
  }

  function normalizeImportedConfig(kind, source) {
    const config = typeof structuredClone === 'function' ? structuredClone(source) : JSON.parse(JSON.stringify(source));
    if (kind === 'simenv') {
      const mass = config.body?.mass;
      if (mass && !mass.randomization) mass.randomization = { distribution: 'none' };
      return config;
    }
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
    if (!['simenv', 'test', 'controller'].includes(kind)) throw new Error('无法判断配置类型，请选择 SimEnv、Controller 或 Train/controller-test 配置。');
    if (kind === 'controller') {
      const config = { controller: source };
      let applied = 0;
      let defaulted = 0;
      document.querySelectorAll('[data-config="test"][data-path^="controller."]').forEach(input => {
        const value = getPath(config, input.dataset.path);
        if (hasImportedValue(value) && setInputValue(input, value)) applied += 1;
        else {
          setInputValue(input, inputDefault(input));
          defaulted += 1;
        }
      });
      if (!applied) throw new Error('Controller 文件中没有与当前参数面板匹配的字段。');
      latestBundle = null;
      const indicator = $('#dirtyIndicator');
      indicator.classList.add('dirty');
      indicator.querySelector('span').textContent = `已导入 ${sourceLabel} · ${applied} 项，默认补齐 ${defaulted} 项`;
      return { kind, applied, defaulted, total: applied + defaulted };
    }
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
  bindParameterTools();
  refreshParameterState();
  $('#exportConfig').addEventListener('click', () => {
    try {
      const bundle = prepareSimulationStart();
      download(bundle.simenv.filename, bundle.simenv.yaml);
      download(bundle.test.filename, bundle.test.yaml);
    } catch (error) {
      $('#dirtyIndicator').classList.remove('dirty');
      $('#dirtyIndicator').classList.add('invalid');
      $('#dirtyIndicator span').textContent = error.message;
    }
  });

  window.RLFlightConfig = {
    buildBundle,
    prepareSimulationStart,
    restoreDefaults,
    importConfig,
    refreshParameterState,
    getLatestBundle: () => latestBundle
  };
})();
