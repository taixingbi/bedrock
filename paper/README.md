建议做两层实验

主实验：容量公平

Bedrock：1 model copy
ECS：1 × g5.xlarge / 1 × A10G

测量：

最大稳定并发
最大稳定 RPS
TTFT P50/P95/P99
总延迟 P50/P95/P99
output tokens/sec
429 / throttling 比例
每 1,000 请求成本
每 1M output tokens 成本

扩展实验：成本公平

固定预算，例如每个平台都花 $10，然后比较：

$10 可以处理多少请求
$10 可以生成多少 tokens
在 P95 latency SLA 下能完成多少请求

这个比单纯比较“每小时多少钱”更有论文价值，因为 Bedrock 和 ECS 的底层硬件并不透明，直接声称硬件完全公平不严谨。