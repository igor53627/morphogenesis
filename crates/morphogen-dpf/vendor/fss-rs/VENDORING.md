# Vendored fss-rs v0.6.0

## Why Vendored

This is a vendored copy of [fss-rs v0.6.0](https://github.com/myl7/fss) (Apache-2.0 license).

We vendor this crate to enable modifications for streaming/chunked DPF evaluation:

1. **Memory optimization**: The upstream `full_eval()` API requires a pre-allocated `&mut [&mut G]` buffer.
   At 25-bit domain (27M pages for Ethereum mainnet), this is 528MB of allocation per query.

2. **Streaming eval API**: We will add `eval_range(start, &mut [G])` and `full_eval_callback(FnMut(usize, G))`
   to enable chunked processing without O(N) allocation.

3. **Zero-copy integration**: Direct accumulation into page-scan loop without intermediate buffers.

## Modifications

None yet - initial vendor. Future phases will add streaming eval API.

## Upstream Compatibility

Based on fss-rs v0.6.0 (commit 8f8f8f8). Any upstream security patches should be manually merged.

## License

Apache-2.0 (same as upstream). See LICENSE file.
