---
library_name: pytorch
tags:
  - chess
  - connectome
  - computational-neuroscience
  - flywire
  - malecns
  - sparse-neural-network
license: other
---

# Flychess ChessFly adapter

This repository contains the released PyTorch/SafeTensors model artifact used
by the Flychess compatibility runtime. The model maps a canonical chess-board
feature vector to policy and value logits through a connectome-shaped recurrent
neural computation. Stockfish is an external opponent and evaluator; it is not
part of these weights.

> [!IMPORTANT]
> This is an engineering compatibility artifact, not a biological MaleCNS or
> FlyWire brain model. It does not establish biological equivalence,
> consciousness, or chess skill in a living fly. The current Flychess
> calibration receipt marks the biological target fit as synthetic-only unless
> a future release explicitly records real held-out targets.

## Model details

- **Format:** PyTorch tensors serialized with SafeTensors.
- **Primary file:** `flynet.safetensors`.
- **Input contract:** 780 canonical board features.
- **Action contract:** 1,968 policy logits representing the public ChessFly
  action space.
- **Value contract:** 64 value-bin logits.
- **Recurrent steps:** five bounded synchronous graph updates.
- **Graph dimensions:** 138,639 neurons and 15,091,983 signed edges.
- **Ports:** 10,855 encoder destinations and 41,692 readout neurons.

The included weights are the public ChessFly/FlyWire-compatible artifact used
as a reference by Flychess. This release does not claim authorship of the
upstream training process; it adds a reproducible release layout, metadata, and
verification receipts around the model file.

## Intended use

This artifact is intended for:

- reproducible research into connectome-shaped neural interfaces;
- educational chess and computational-neuroscience demonstrations;
- compatibility testing of the Flychess adapter and its legal-move boundary;
- controlled experiments comparing neural readouts with an external chess
  engine.

Do not use it for safety-critical decisions, claims about animal cognition, or
as a general chess engine. No Elo, playing-strength, or biological-performance
claim is made by this repository.

## Architecture and tensor contract

The runtime applies the following computation to a white-perspective board
encoding:

```text
drive = encoder(board_features)
h = 0
for t in range(5):
    recurrent = W @ h
    pre = (recurrent + drive) * scale[t] + shift[t]
    h = (1 - alpha) * h + alpha * relu(pre)
association = GELU(decoder(readout(h)))
policy = policy_head(association)
value = value_head(association)
```

The released tensor shapes are:

| Tensor | Shape |
| --- | --- |
| `encoder.weight` | `[10855, 780]` |
| `encoder.bias` | `[10855]` |
| `scale`, `shift` | `[5, 138639]` |
| `log_gain` | `[15091983]` |
| `decoder.weight` | `[512, 41692]` |
| `decoder.bias` | `[512]` |
| `policy.weight` | `[1968, 512]` |
| `policy.bias` | `[1968]` |
| `value.weight` | `[64, 512]` |
| `value.bias` | `[64]` |

For black-to-move positions the adapter mirrors into a white perspective and
maps candidate moves back. Legal-move masking is deliberately applied outside
the neural graph by the chess policy layer.

## Repository contents

This is a model repository, not a mirror of the Flychess source repository.

- `flynet.safetensors` — the actual model weights;
- `config.json` — architecture, dimensions, and compatibility metadata;
- `model_index.json` — model artifact index metadata;
- `results/` — release manifest, checksums, and verification receipts;
- `README.md` — this model card.

The large companion graph binaries are distributed separately so consumers can
choose their download and licensing path. The release manifest records the
expected graph sources and hashes used for compatibility checks.

## Quick start

Install the runtime and Hub client in an isolated environment, then download
the model file and the companion graph artifacts:

```bash
~/.venv/bin/python -m pip install 'huggingface_hub' 'safetensors' 'torch' 'python-chess'
```

```python
import os
from huggingface_hub import hf_hub_download

model_repo = os.environ["FLYCHESS_MODEL_REPO"]
graph_repo = os.environ["FLYCHESS_GRAPH_REPO"]

weights_path = hf_hub_download(
    repo_id=model_repo,
    filename="flynet.safetensors",
    repo_type="model",
)
connectome_path = hf_hub_download(
    repo_id=graph_repo,
    filename="data/connectome.bin.gz",
    repo_type="space",
)
neurons_path = hf_hub_download(
    repo_id=graph_repo,
    filename="data/neurons.bin.gz",
    repo_type="space",
)

from flychess.chessfly import ChessFlyModel, ChessFlyPolicy

model = ChessFlyModel.from_artifacts(
    connectome_path,
    neurons_path,
    weights_path,
    device="cpu",
)
policy = ChessFlyPolicy(model)
move = policy.select_move(board)
print(move)
```

Set `FLYCHESS_MODEL_REPO` to this model repository and
`FLYCHESS_GRAPH_REPO` to the companion graph release. Always verify the
downloaded files against `results/release.json` before running an experiment.

## Evaluation and verification

The release receipts cover safe tensor loading, tensor-key and shape checks,
graph/header compatibility, the forward contract, legal-move masking, and the
Flychess test suite. These are integration checks, not a chess-strength
benchmark. The exact commit, artifact sizes, SHA-256 digests, test command, and
biological-calibration status are recorded in `results/release.json`.

## Data, provenance, and biological calibration

The model and graph have distinct provenance. The upstream public ChessFly
model and demo provide the compatibility reference; FlyWire/MaleCNS data terms
and source papers remain authoritative for graph-derived artifacts. Flychess
keeps MaleCNS and FlyWire as separate source families and records node-ID
namespace, graph version, sign policy, and target status in calibration
manifests. A connectome-shaped computation should not be described as a
calibrated biological brain without measured, held-out functional validation.

## License and attribution

The metadata license is `other` because upstream model and graph terms govern
the included and referenced artifacts. Review the upstream terms before
redistribution or commercial use; this card grants no additional rights.

Reference implementations:

- [ChessFly model](https://huggingface.co/mlabonne/chessfly)
- [ChessFly demo Space](https://huggingface.co/spaces/mlabonne/chessfly)
- [FlyWire](https://flywire.ai/)

