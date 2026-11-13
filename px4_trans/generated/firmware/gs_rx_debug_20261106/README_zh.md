# 飞控内部GS-Link RX诊断

固件：micoair_h743-v2_nncontrol_gs_rx_debug.px4。已完成目标板编译和GS-Link主机协议检查，未刷入实体机。配套ELF与固件来自同次构建。

本次仅增加GS-Link接收诊断及修正mocap年龄显示下溢；保留当前工作区既有控制器、EKF与日志设置。没有调整波特率、融合门限、时间同步门限或已保存参数。

## 为什么需要它

旧status中的rx_packets只统计完整CRC通过的帧。rx_packets=0、CRC错误=0不能区分没有字节、原始字节无帧头、read报错或解析器一直等待残帧。因此有物理波形后，下一步应观察驱动和用户态的字节计数。

## 不刷机时先核对

在未解锁、使用USB/QGC控制台的状态下执行：

```sh
gs_link status
mavlink status
airlink status
param show MAV_0_CONFIG
param show MAV_1_CONFIG
param show MAV_2_CONFIG
param show SER_TEL1_BAUD
```

确认没有其他模块也使用/dev/ttyS0。此处只查询，不应盲目停止其他飞行模块。TELEM1波特率应与网桥一致，为4000000；gs_link的启动-b参数也决定其实际配置。

本板构建的/dev/ttyS0为USART1，RX=PA10，TX=PA9。依据板级board.h和STM32H7串口设备注册顺序核对，不能把其他串口上的波形当成USART1输入。

## 刷入诊断固件后

保留地面站运行，在未解锁状态下：

```sh
gs_link status
sleep 3
gs_link status
```

新增状态：

- RX read calls/bytes：串口读取调用次数、read实际返回的字节总数。
- empty/eagain/errors、errno：零字节返回、非阻塞暂时无数据、真正读取错误和最近错误号。大量eagain本身正常。
- RX pending last/max：拥有串口描述符的工作队列中用FIONREAD采样的驱动待收字节。last经常为0正常，结合bytes增长判断；max表示曾看到的峰值。
- parser buffered/discarded/overflows：解析缓冲残留、非帧头丢弃字节及缓冲溢出计数。
- RX first/last bytes：首个读块开头与最新读块末尾最多32字节十六进制样本。只在status调用时打印，不逐字节刷日志。
- heartbeats、sync requests/replies、unknown DATA：有效心跳、有效时间同步请求/成功回复和未处理DATA数量。

这些是累计计数；两次status差值最有意义。rx_bytes为32位计数，长时间高流量可回绕。状态打印与工作队列并发，原始hex可能跨读块更新，仅用于定位层次，不作为完整抓包证据。

| 现象 | 优先检查 |
|---|---|
| read calls不增长 | 模块工作队列是否调度、串口是否打开 |
| calls增长、bytes=0、pending max=0、只有eagain | UART驱动未交付数据，或其他读取者已经取走；检查USART1引脚复用、RX使能/中断、端口占用 |
| read errors增长 | 按errno检查fd/任务组/设备错误；NuttX fd应由工作队列持有，勿跨任务借用 |
| bytes增长、discarded增长、无有效帧 | 查原始hex是否有A5 5A；核对4M、8N1、极性、波形能否实际解码 |
| bytes增长、CRC/version/length错误增长 | 字节损坏、错协议、参数/帧边界问题 |
| parser buffered长时间不变且无完整帧 | 看收到的头和声明长度，是否停在不完整帧 |
| 有心跳、无sync requests | 时间同步请求在上游未到达，或ID/flags/长度不符 |
| sync requests增长、replies不增长 | 回复write失败，查write failures/errno |
| requests/replies均增长、地面站received仍为0 | 查网桥回传、UDP接收端/解析；若received增长，再看地面站拒绝原因 |

正常同步请求开头应包含`a5 5a 01 01 04 00`，payload长度字段为`08 00`；心跳开头为`a5 5a 01 02 02`，payload长度`10 00`。seq、时间戳和CRC随帧变化。

不额外启动cat/串口读取程序与gs_link竞争同一RX队列。若要独立原始串口测试，应在未解锁状态先停止gs_link，且由测试程序显式配置串口。

## 字节仍为0时：SWD/GDB进一步检查

需要调试探针与本次ELF，且只用于不飞行的地面诊断。当前构建USART1没有RX DMA，使用中断接收。

关注源码`platforms/nuttx/NuttX/nuttx/arch/arm/src/stm32h7/stm32_serial.c`：

1. 在`up_interrupt`设置条件断点，条件为`arg == g_uart_devs[0]`（USART1）。若有外部输入却从不触发，检查GPIO/USART/NVIC配置。
2. 查看USART1状态中的RXNE/RXFNE、ORE、FE、NE，以及CR1的UE、RE、RXNEIE使能。使用本源码的寄存器宏解释，不盲目改写寄存器。
3. 跟踪`uart_recvchars`将RDR字节搬入NuttX软件接收队列的路径，比较队列head/tail和用户态read；区分硬件接收、中断服务、软件队列和应用读取。

不要在诊断中反复读取USART RDR数据寄存器：读取会消耗数据，可能改变原本现象。断点会暂停调度并影响链路超时，因此该步骤只用于UART接收机制定位。
