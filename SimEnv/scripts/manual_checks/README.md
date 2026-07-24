# SimEnv 手工动力学检查

这些脚本使用由 `configs/example.yaml` 派生的临时、确定性配置，把噪声、参数随机化、
回差、死区和格栅衰减等干扰项关闭，再逐项打印并断言可手算的中间结果。临时配置和
日志随进程退出删除，不会修改训练配置。

一次运行全部检查：

```bash
python scripts/manual_checks/run_all.py
```

也可以单独运行：

```bash
python scripts/manual_checks/check_configured_geometry.py configs/example.yaml
python scripts/manual_checks/check_control_surface_moments.py
python scripts/manual_checks/check_time_integration.py
python scripts/manual_checks/check_observations.py
```

`check_configured_geometry.py` 不替换几何参数，可同时审计任意多个实际配置：

```bash
python scripts/manual_checks/check_configured_geometry.py \
  configs/example.yaml \
  ../Train/configs/environment/mlp_nominal_baseline_1.yaml
```

坐标系采用 NED/FRD。机体系 `+z` 指向下方；从机体上方向下看（视线沿 `+z`），
右手定则的 `+Mz` 是顺时针偏航。舵面脚本中的 `all +` 表示三个舵面都绕各自的径向
偏转轴按右手正方向偏转，因此预期三个偏航力矩同号叠加为 `+Mz`。

脚本失败时会抛出断言并显示实际值和手算值。观测脚本还会确认自由落体初始比力为
零，并在旋转工况下直接比较加速度计输出与同一物理步的 `force_b / mass`。
