# flychess

Flychess is a reproducible research project for sparse recurrent graph policies
that play chess and are evaluated against [Stockfish](https://stockfishchess.org/).

## Current status

The source repository now contains an independent FlyNet track:

- a seeded sparse signed graph generator with explicit optic, association,
  memory, action, and value regions;
- an 851-feature board encoder and deterministic 4,544-action UCI vocabulary;
- a trainable recurrent graph model with inspectable activity readouts;
- a legal-position sampler with side-to-move Stockfish teacher labels;
- reproducible training, validation, SafeTensors export, graph serialization,
  checksums, and release manifests;
- CLI and local web support for loading a trained FlyNet release;
- a Colab-oriented workflow for generating data, training, evaluating, and
  publishing a model-only artifact;
- tests covering graph/dataset contracts, checksum validation, game plumbing,
  replay integrity, and HTTP boundaries.

The first independent FlyNet release was trained from a seeded random
initialization in a Colab T4 runtime and published as a model-only artifact.
The release includes the SafeTensors weights, generated graph, configuration,
training history, and SHA-256 release receipt. Its reported validation metrics
are 20.84% top-1 teacher agreement, 45.34% top-5 teacher agreement, and
0.3066 value mean absolute error on 5,000 held-out Stockfish-labelled
positions. These are teacher-agreement and value-prediction measurements, not
an Elo rating or evidence of human-level playing strength.

The current release was trained on 500,000 Stockfish-labelled positions for
three epochs with a 1,024-node graph, 32,000 signed directed edges, and three
recurrent graph updates. The model repository contains the model artifacts;
the source repository contains the implementation and verification tooling.

FlyNet's graph is an engineering abstraction, not a biological connectome
reconstruction. MaleCNS/FlyWire calibration remains a separate provenance-
locked research track. The legacy artifact adapter is retained only as a
clearly isolated comparison path in [the comparison runtime note](docs/chessfly-runtime.md).

## Run the source checks

Use the shared Python environment required by this workspace:

```bash
~/.venv/bin/python -m pip install -e '.[dev]'
~/.venv/bin/python -m pytest -q
```

Run the existing dependency-light baseline locally:

```bash
~/.venv/bin/flychess --depth 3 --max-plies 40
```

Serve the browser UI:

```bash
~/.venv/bin/python -m flychess.web
```

The server binds to `127.0.0.1` by default. The browser follows a bounded run
through the live state endpoint. The Fly decision panel shows surfaced legal
candidates and compact activity metrics; this is observable policy telemetry,
not a literal private chain-of-thought or evidence of biological cognition.

## Train FlyNet in Colab

The checked-in notebook [notebooks/flynet_colab.ipynb](notebooks/flynet_colab.ipynb)
contains the complete hosted-runtime sequence. The equivalent cells are:

```bash
%cd /content
!git clone https://github.com/EF-Code/flychess.git
%cd /content/flychess
!python -m pip install -q -e '.[flynet,dev]'
!sudo apt-get update -qq
!sudo apt-get install -y -qq stockfish
```

Generate an auditable teacher dataset. This uses Stockfish for labels only;
it does not load an existing neural model:

```bash
!python scripts/generate_flynet_dataset.py \
  --engine stockfish \
  --depth 3 \
  --samples 20000 \
  --seed 20260917 \
  --output /content/flynet-dataset.npz
```

Train a small baseline from a seeded random initialization and write a complete
release bundle. The command below is a local or Colab example; it is not the
configuration used for the published 500,000-position release:

```bash
!python scripts/train_flynet.py \
  --dataset /content/flynet-dataset.npz \
  --output-dir /content/flynet-release \
  --seed 20260917 \
  --graph-seed 20260918 \
  --nodes 4096 \
  --edges 200000 \
  --epochs 20 \
  --batch-size 256 \
  --device cuda
```

Evaluate the held-out teacher agreement and legal-move boundary:

```bash
!python scripts/evaluate_flynet.py \
  --dataset /content/flynet-dataset.npz \
  --weights /content/flynet-release/flynet.safetensors \
  --graph /content/flynet-release/flynet-graph.npz \
  --graph-metadata /content/flynet-release/flynet-graph.json \
  --config /content/flynet-release/flynet-config.json
```

Play the trained model locally:

```bash
!python -m flychess.cli \
  --flynet-weights /content/flynet-release/flynet.safetensors \
  --flynet-graph /content/flynet-release/flynet-graph.npz \
  --flynet-graph-metadata /content/flynet-release/flynet-graph.json \
  --flynet-config /content/flynet-release/flynet-config.json \
  --engine stockfish --depth 3 --max-plies 40
```

## Publish the independent model artifact

The GitHub source distribution and Hugging Face model distribution are
separate. The model repository contains the actual SafeTensors weights, the
generated sparse graph, config, optional labeled dataset, training history,
model card, and release receipt. It does not mirror the source tree.

After the source commit has been pushed and the release bundle passes local
validation, publish from Colab using the `HF_TOKEN` runtime secret. A
`GITHUB_ACCESS_TOKEN` is only required when the source checkout cannot be
read anonymously.

```bash
!python scripts/publish_flynet_model.py \
  --source-dir /content/flychess \
  --release-dir /content/flynet-release \
  --dataset /content/flynet-dataset.npz \
  --repo-id YOUR_HF_NAMESPACE/flychess \
  --prune
```

The publisher refuses dirty or stale source checkouts, private home-path
references, malformed graph/config/weights, missing random-initialization
provenance, failed tests, or an unexpected final Hub file set. `--prune` is
explicit because it removes stale files outside the new model-only payload.

## Measuring “100× better”

“100× better” is a target, not a result that can be assumed from a larger
graph. The comparison harness must use the same fixed FEN suite, Stockfish
depth, time budget, hardware, and color balance for every candidate. It should
report at least:

- legal-move rate;
- top-1 and top-5 teacher agreement;
- centipawn loss against the teacher;
- p50/p95 decision latency and memory use;
- head-to-head game score over a seeded suite;
- ablations for graph size, recurrent steps, dataset size, and teacher depth.

Only a release receipt containing those measurements can support a claim of
improvement. Training loss or a single demonstration is not sufficient.

## Other experiment modes

Record a replayable baseline game:

```bash
~/.venv/bin/flychess --depth 3 --max-plies 40 \
  --record runs/game.jsonl --result runs/result.json
```

Run the small JSON edge-list backend:

```bash
~/.venv/bin/flychess --connectome examples/tiny-connectome.json \
  --neural-steps 2 --depth 2 --max-plies 24
```

The comparison adapter and its artifact-specific assumptions are documented
separately so they cannot be mistaken for FlyNet provenance.

## Scientific boundary

Flychess uses precise language: a simulated graph is not a living fly, and a
game-playing model is not evidence of consciousness or biological equivalence.
Any future MaleCNS/FlyWire calibration must preserve source identity, node-ID
namespace, graph version, sign policy, held-out functional targets, and the
boundary between measured data and engineering assumptions.
