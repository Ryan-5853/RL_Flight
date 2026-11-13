# USART1中断与接收队列诊断

固件：micoair_h743-v2_nncontrol_gs_uart_driver_debug.px4。配套ELF来自同次构建。
已编译、检查符号和诊断字符串，未刷机、未在实体机执行验证。

之前的read calls只统计应用轮询，不是中断次数。本版本通过NuttX配置STM32H7_USART1_RXDIAG启用USART1专用计数，只适用于当前非RX DMA接收路径。该选项默认为关闭，当前H743-v2诊断构建开启。

## 操作

在未解锁状态刷入，保留地面站运行，通过USB/QGC执行：

```sh
gs_link status
sleep 3
gs_link status
```

应看到标识`UART RX diagnostic revision: ISR-QUEUE-1`，以及新增字段：

```text
USART1 IRQ/RXNE/service/RDR/queued: ...
USART1 errors FE/NE/ORE: ..., queue-full events: ...
USART1 driver IE/last RX ISR: ..., queue head/tail/size: ...
USART1 driver baud/kernel clock: ...
```

计数含义：

- IRQ：真实USART1中断处理函数执行次数，包含发送与接收，不能单独证明接收。
- RXNE：USART1中断入口检测到RXNE/RXFNE的次数，表示当时硬件接收寄存器/FIFO非空。
- service：接收分支调用uart_recvchars的次数。
- RDR：驱动原有up_receive函数实际读取RDR的字节数，没有新增数据寄存器读取。
- queued：uart_recvchars执行前后软件环形队列head的前移字节数，按环长处理回绕。UART1 ISR是此非DMA队列的生产者；线程不能在ISR中运行。队列容量为size-1，满队列后丢弃的字节不会计入queued。
- FE/NE/ORE：中断入口观察到的帧错误、噪声、硬件溢出事件次数，不是坏字节的精确数量；同一粘滞标志可能被多次观察。
- queue-full events：接收服务返回时观察到软件环形队列已满的次数，不是丢失字节总数。
- driver IE：NuttX驱动缓存的中断使能，需与实际CR1结合检查；last RX ISR保存最近看到RXNE时的状态，补足瞬时status漏掉接收状态的问题。
- queue head/tail/size：接收环形缓冲的写入/读取索引及大小。该驱动快照在短临界区内取得；应用read计数是在稍后打印的，不能要求两者同一瞬间相等。

所有计数为开机以来累计值，32位长期运行可能回绕，重点比较两次status的增量。没有在ISR中printf、没有改UART参数、没有额外读RDR或写ICR。

## 判读

| 增量模式 | 定位方向 |
|---|---|
| IRQ增长、RXNE/RDR/queued/read bytes都为0 | 中断可能仅来自TX；硬件没有向当前ISR呈现接收数据。结合PA10波形、GPIO复用、RX使能与错误事件检查 |
| RXNE增长、service为0 | 驱动接收分支没有执行，检查driver IE与CR1 RXIE是否一致，以及中断分支条件 |
| service增长、RDR为0 | uart_recvchars没有取到硬件字节；检查rxavailable/FIFO状态和接收路径 |
| RDR增长、queued很少且queue-full增长 | 数据从硬件取出，但软件队列满导致丢弃；看head/tail是否前进 |
| queued增长、应用read bytes为0 | 软件队列确有数据；排查其他读取者、串口设备映射和应用fd/读取路径 |
| FE/NE持续增长 | UART确实观察到异常信号，重点检查位宽、电平、时序、极性及实际4 Mbps解码 |
| ORE增长 | 硬件接收服务不及时或长时间屏蔽；不能直接归因物理连线 |

配合此前的GPIO、USART、NVIC快照，可进一步缩小故障层次，但软件计数本身不能唯一证明芯片损坏。若所有配置正确而RXNE始终为0，仍需到MCU PA10确认波形能按4M/8N1正确解码，或以本地回环和正常板对照验证。

## 修改范围

- NuttX stm32h7/Kconfig、stm32_serial.c和新增stm32_usart1_rxdiag.h。
- H743-v2 nsh/defconfig启用诊断。
- gs_link status打印驱动快照。

未修改控制器模型、EKF融合算法或任何已保存参数；新增计数有少量运行开销，本镜像用于地面接收故障诊断。
