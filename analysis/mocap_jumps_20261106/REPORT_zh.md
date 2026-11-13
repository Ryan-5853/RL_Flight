# 晃动飞机时的姿态跳变诊断

源日志：log/26.11.6/log_0_2026-11-6-10-54-50.ulg。以下时间均为飞控启动后秒数。

结论：确认一次启动yaw重置；无法根据这份日志确认晃动过程存在真实EKF不连续跳变，也无法排除发生在日志缺失区间的异常。

- 1.954496 s：姿态reset counter由0变1，yaw约0.56°→17.28°，重置旋转16.72075°。随后到47.387 s，counter一直为1。
- estimator_status外部航向标志约2.036 s起为true，之后记录中无停融切换；这是约5 Hz采样的融合意图标志，不等于每条外部观测均成功更新。
- 对连续且相邻间隔小于10 ms的数据，用机载角速度预测姿态增量，与实际四元数增量比较。5 s之后差异99分位0.090°，最大0.152°。未见这部分数据中额外的多度瞬间跳变。该检验是局部运动一致性检查，不是绝对姿态精度检验。
- 姿态日志42个间隔超过30 ms，最大411.85 ms。ULog共118条dropout记录，累计7.573 s。角速度/FIFO等多路在同一时间附近也缺失。把缺口两侧直接连线，会出现很大的可见跨步，但不能解释为单个滤波更新。
- 例如36.837204→37.204032 s，中间366.828 ms未录到姿态；两端roll从32.64°变为-36.78°。39.210082→39.581952 s类似，间隔371.870 ms，roll从38.59°到-34.49°。这些区间不参与连续数据的跳变判断。
- 本日志没有vehicle_visual_odometry、estimator_aid_src_ev_att、estimator_status_flags。不能直接比较动捕四元数是否跳变、时间同步是否漂移，也不能检查逐帧创新与拒绝情况。
- 代码复查发现实际initialize_logged_topics使用定制的高频辨识主题列表，绕过默认主题集；之前在默认主题集追加动捕日志没有覆盖实际路径。这是前次日志接入的遗漏。
- EKF2_DELAY_MAX=200 ms，EV_CTRL=16，EV_DELAY=0，EVA_NOISE=0.05，MAG_TYPE=5。200 ms融合延迟本身不是无线链路延迟。
- 滤波故障位始终0，未解锁。加速度计Z偏置很早达到-0.4 m/s²限幅，日志多次High Accelerometer Bias；这是另一个需要检查的异常，不能仅由本日志归因于动捕融合。

下一次需要在实际定制主题列表记录：vehicle_attitude(200 Hz)、vehicle_angular_velocity(200 Hz)、vehicle_visual_odometry(50 Hz)、estimator_aid_src_ev_att(50 Hz)、estimator_status_flags、estimator_sensor_bias，并减少FIFO日志压力。QGC显示跳变还需要结合接收间隔或遥测记录，区分链路/显示刷新与EKF输出异常。

图：attitude_diagnostics.png。灰色为日志缺口，红线为启动yaw重置；图中姿态线在缺口处断开，避免制造伪跳变。
