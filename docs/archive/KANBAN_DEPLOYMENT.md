# Kanban: Deployment & Integration Testing

**Objective:** Deploy the finalized 48-byte Mainnet Matrix to the Modal cloud environment and validate the end-to-end Verifiable PIR pipeline.

## 📋 Backlog

### 📦 Data Migration
- [ ] **Transfer Dataset:** Move `mainnet_optimized48.bin` (97 GB) from `hsiao` (`/root/morphogenesis/`) to Modal Volume `morphogen-data`.
    - *Method:* SSH into `hsiao`, authenticate `modal`, run `modal volume put morphogen-data mainnet_optimized48.bin /db.bin`.
- [ ] **Verify Integrity:** Run a checksum on `hsiao` and compare with the uploaded file on Modal (via a Modal function).

### 🧪 Verification Testing
- [ ] **Run Full Benchmark:** Execute `modal_fused_bench.py` against the real `/db.bin` (97GB).
    - *Success Criteria:* Latency < 50ms, Throughput > 2.0 TB/s on H200.
    - *Check:* Verify that `verif0..2` outputs are non-zero and consistent.
- [ ] **End-to-End Client Test:**
    - Deploy `morphogen-server` on Modal using the new dataset.
    - Run `morphogen-client` (local or remote) to query a known account (e.g. `0xd8dA...`).
    - **Verify:** Client must successfully validate the Sum-Check proof returned by the server.

## 🏗️ In Progress
*None*

## ✅ Done
- [x] **Data Generation:** Generated `mainnet_optimized48.bin` on `hsiao`.
- [x] **Code Readiness:** Client/Server/Core updated to support Binius proofs and 48B schema.
