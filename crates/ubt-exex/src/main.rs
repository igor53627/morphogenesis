use futures_util::StreamExt;
use reth_exex::{ExExContext, ExExEvent, ExExNotification};
use reth_node_api::FullNodeComponents;
use std::sync::Arc;

#[cfg(feature = "reth")]
pub async fn exex_init<Node: FullNodeComponents>(
    ctx: ExExContext<Node>,
) -> eyre::Result<impl futures_util::Future<Output = eyre::Result<()>>> {
    Ok(exex_run(ctx))
}

#[cfg(feature = "reth")]
pub async fn exex_run<Node: FullNodeComponents>(
    mut ctx: ExExContext<Node>,
) -> eyre::Result<()> {
    while let Some(notification) = ctx.notifications.next().await {
        match notification {
            ExExNotification::ChainCommitted { new } => {
                println!("Chain committed: block {}", new.tip().number);
                // 1. Extract changed accounts
                // 2. Map to Cuckoo indices
                // 3. Send XOR diffs to morphogen-server
            }
            ExExNotification::ChainReorged { old, new } => {
                println!("Chain reorged: from {} to {}", old.tip().number, new.tip().number);
            }
            ExExNotification::ChainReverted { old } => {
                println!("Chain reverted: block {}", old.tip().number);
            }
        }
    }
    Ok(())
}

fn main() {
    println!("=== Morphogenesis UBT ExEx ===");
    println!("This binary is intended to be run as part of a Reth node.");
    println!("Compile with --features reth to enable integration.");
}
