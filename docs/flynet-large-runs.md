# Large FlyNet training runs

The normal FlyNet dataset helper keeps every row in memory. Large runs use the
sharded pipeline instead:

1. Stockfish labels are written to independently compressed `float16` shards.
2. Each shard has a JSON record containing its global index range, storage
   dtypes, byte count, and SHA-256 digest.
3. `manifest.json` is updated atomically after every completed shard.
4. Training streams one shard at a time and writes a local checkpoint after
   every completed epoch.

This makes runtime interruption recoverable without trusting an incomplete
`.npz` file. The model remains a from-scratch FlyNet artifact; Stockfish is
only the move/value teacher.

## Capacity planning

A 50M-label run means 50 million labelled positions, not just 50 million
game plies. The live T4 Colab probe on 2026-09-17 measured approximately
57 labels/sec at depth 3 and 66 labels/sec at depth 2. That is roughly 212–243
hours for one CPU-bound runtime, before restarts. Multiple Stockfish workers did
not improve this particular runtime, so the job should be split across durable
shard ranges or separate runtimes rather than launched as one blocking cell.

Keeping the original float32 feature matrix would require about 170 GB before
FENs and compression. Shards store the encoder output as float16 and retain
the FENs, labels, and checksums. A 100,000-row shard is the default balance
between restart granularity and compression overhead.

## Colab commands

Run the generator with a modest pilot first:

```bash
python scripts/generate_flynet_shards.py \
  --root /content/flynet-shards-1m \
  --samples 1000000 \
  --shard-size 100000 \
  --depth 2 \
  --workers 1
```

After the manifest is complete, stream it through a scratch model:

```bash
python scripts/train_flynet_sharded.py \
  --dataset-root /content/flynet-shards-1m \
  --output-dir /content/flynet-1m-release \
  --epochs 1 \
  --batch-size 512 \
  --device cuda
```

For a 50M target, reserve a deterministic validation fraction using the
global-index modulus. Do not interpret the first pilot's score as playing
strength: report legal rate, teacher top-k agreement, held-out value error,
latency, and fixed Stockfish-game outcomes together.
