# Morphogenesis PIR - QPS Sub-Kanban

## [GOALS]
- Increase QPS while preserving 2-server DPF privacy model
- Keep per-query tail latency stable under load
- Maintain GPU HBM residency of full dataset

## [IDEAS / BACKLOG]

### Batching
- [ ] Multi-query batching per scan (evaluate many DPF keys in one pass)
- [ ] Server-side batching window (N ms) with admission control
- [ ] Client-side batching API (submit multiple keys at once)
- [ ] Batch-aware response framing (streamed partials to reduce tail)

### Multi-GPU / Sharding
- [ ] Shard DB across GPUs and aggregate responses (per-server)
- [ ] Inter-GPU reduction strategy (NVLink, host-staged XOR)
- [ ] GPU scheduler for load balancing across shards
- [ ] Horizontal scaling across hosts (consistent shard map)

### Kernel / Memory Bandwidth
- [ ] Tune vectorized load width per GPU architecture
- [ ] Persistent kernel with double-buffered DPF masks
- [ ] Minimize atomics in reduction (warp-level reduction then block)
- [ ] Ensure fully coalesced reads for page layout

### Page / Entry Layout
- [ ] Evaluate smaller entry size (32B vs 40B) impact on scan BW
- [ ] Page size sweep (2KB/4KB/8KB) for HBM efficiency
- [ ] Alignment and padding audit for zero waste

### Query Parallelism
- [ ] Parallel scan streams per GPU (overlap compute+copy)
- [ ] Pipelined PRG expansion vs scan vs reduction
- [ ] CPU-side pre-expansion of DPF seeds where possible

### Cuckoo / Addressing
- [ ] Evaluate κ=2 vs κ=3 tradeoff under real load
- [ ] Stash pressure metrics and rehash thresholds
- [ ] Seeds versioning + rolling rehash strategy

### Network / Transport
- [ ] Binary wire format for DPF keys and responses
- [ ] HTTP/2 or QUIC multiplexing to reduce per-request overhead
- [ ] Backpressure and request queueing policy

### Observability
- [ ] End-to-end QPS dashboard (per GPU, per shard)
- [ ] Breakdown: DPF eval vs scan vs reduction vs network
- [ ] Tail latency tracking at p95/p99

## [CANDIDATE EXPERIMENTS]
- [ ] Batch size sweep (1, 4, 8, 16, 32 keys/scan) vs QPS
- [ ] 2-GPU sharding test with XOR aggregation
- [ ] Page size A/B test on H100/H200
- [ ] κ=2 vs κ=3 throughput comparison

## [OPEN QUESTIONS]
- [ ] Target QPS per GPU for production assumptions
- [ ] Acceptable batching latency window (ms)
- [ ] Multi-tenant fairness requirements
