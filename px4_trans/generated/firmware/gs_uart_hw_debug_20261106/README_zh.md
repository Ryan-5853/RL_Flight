# USART1硬件接收配置诊断

已编译，未刷机。沿用上一版gs_rx_debug的字节/解析计数；新增gs_link status中的只读硬件快照。不改控制、融合、串口配置，不读取RDR、不写ICR、不清除错误标志。

目标仅为micoair_h743-v2、/dev/ttyS0（USART1）。地址和位号按当前NuttX STM32H743头文件核对：USART1基址0x40011000，GPIOA基址0x58020000，USART1外部IRQ37。

在未解锁状态刷入后，仅需：

```sh
gs_link status
sleep 3
gs_link status
```

预期：

- PA10 mode=2、AF=7：PA10处于USART1所需的复用模式；pull=1为上拉。input是瞬时电平，串口空闲通常高，不能靠一次input=1认定没有波形。
- UE=1、RE=1：USART和接收器使能。
- RXIE=1：当前非DMA构建启用了RX非空中断。
- SWAP=0、RXINV=0、HDSEL=0、DMAR=0：正常交叉连线、非反相、全双工、中断接收。先确认实际接线/协议，再判断非默认值是否错误。
- IRQ37 EN=1：NVIC中USART1中断使能。
- PEND/ACTIVE是瞬时快照，正常情况下经常为0，不能代替中断执行次数计数。
- RXNE/RXFNE持续1且应用bytes持续0：沿USART中断、NVIC、NuttX接收队列查。一次0不能证明UART未收到，因为中断可能刚处理完。
- ORE/FE/NE：溢出、帧错误、噪声。错误位可能被驱动正常清除，单次0不等于整个运行期无错误。

如果GPIO复用正确、RX使能/中断开启，但应用仍无字节，下一步应在USART1中断处理与uart_recvchars中加计数或使用SWD/GDB跟踪，不能直接判定硬件损坏。

MAVLink在ttyS2/USB和AIRLINK在ttyS1的状态已排除这两个模块占用ttyS0；尚不能排除其他未列出的读取者。不要在gs_link运行时启动第二个串口读程序争抢数据。

板级端口声明见boards/micoair/h743-v2/default.px4board，PA10/PA9定义见nuttx-config/include/board.h。串口注册顺序见NuttX stm32h7/stm32_serial.c。
