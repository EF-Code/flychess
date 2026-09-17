# flychess

`flychess` is an experiment in connecting a fruit-fly brain model to chess,
with [Stockfish](https://stockfishchess.org/) as the opponent.

## Current status

The repository contains a runnable local experiment:

- a stable sensory contract for an 8x8 board;
- a deterministic recurrent `SurrogateFlyBrain` policy;
- a validated JSON edge-list `Connectome` backend and chess adapter;
- legal-move validation through `python-chess`;
- a real Stockfish UCI opponent;
- a bounded command-line game loop, replay recorder, and local browser UI;
- a bounded fly decision readout showing the top legal candidates and policy activity;
- a provenance-locked MaleCNS calibration importer for filtered subgraphs;
- a sparse sign-aware LIF dynamics backend for imported calibration subgraphs;
- a compatible optional runtime for the public ChessFly/FlyWire artifact;
- tests covering graph validation, replay tampering, HTTP boundaries, and engine integration.

The surrogate is deliberately not presented as a biological fly brain. A
connectome gives us a wiring diagram, but calibrated neural dynamics,
chemical signaling, sensory transduction, and motor semantics still need to
be supplied. The policy interface is designed so a MaleCNS/FlyWire backend can
replace the surrogate without changing the game or engine layers.

## Run it

Use the shared Python environment required by this workspace:

```bash
~/.venv/bin/python -m pip install -e '.[dev]'
~/.venv/bin/flychess --depth 3 --max-plies 40
```

If the console script is not on the environment path, use:

```bash
~/.venv/bin/python -m flychess.cli --depth 3 --max-plies 40
```

Serve the browser UI locally:

```bash
~/.venv/bin/python -m flychess.web
```

The server binds to `127.0.0.1` by default. Open the printed URL and use the
bounded game form to run an experiment.

The browser follows a run live through `POST /api/game/start` and repeated
`GET /api/state` snapshots. The **Fly decision readout** panel surfaces up to
five legal candidates, their policy scores, the selected move, and compact
activity metrics as the game progresses. This is observable policy telemetry,
not a literal private chain-of-thought or evidence of biological cognition.

Record a replayable game:

```bash
~/.venv/bin/flychess --depth 3 --max-plies 40 \
  --record runs/game.jsonl --result runs/result.json
```

Run the graph backend with a connectome edge-list JSON file:

```bash
~/.venv/bin/flychess --connectome examples/tiny-connectome.json \
  --neural-steps 2 --depth 2 --max-plies 24
```

## Run the public ChessFly artifact

The optional ChessFly runtime can load the graph and weights used by the
public demo. The graph is downloaded from the demo Space, while the learned
weights are downloaded from the model repository; neither is vendored in this
repository. Install the isolated runtime explicitly:

```bash
~/.venv/bin/python -m pip install -e '.[chessfly]'
```

Then construct a policy after acquiring and checksum-recording
`connectome.bin.gz`, `neurons.bin.gz`, and `flynet.safetensors`:

```python
from flychess.chessfly import ChessFlyModel, ChessFlyPolicy

model = ChessFlyModel.from_artifacts(
    "data/chessfly/connectome.bin.gz",
    "data/chessfly/neurons.bin.gz",
    "data/chessfly/flynet.safetensors",
)
fly = ChessFlyPolicy(model)
move = fly.select_move(board)
print(move, fly.win_probability, fly.last_readout)
```

The adapter matches the public worker's 780-feature encoding, black-turn
mirroring, 1,968-action space, target-row CSR propagation, five calibrated
steps, decoder heads, and post-readout legal mask. It is a compatibility layer
for a FlyWire-derived model, not biological MaleCNS calibration. Keep the
model's graph license and source citations with any acquired artifacts.

## Publish the model artifact from Colab

The GitHub source distribution and the Hugging Face model distribution are
separate by design. The model repository contains the SafeTensors weights,
model card, architecture metadata, and a release receipt; it does not mirror
the Python source tree. From a Colab checkout with `HF_TOKEN` and
`GITHUB_ACCESS_TOKEN` stored as runtime secrets, run:

```bash
python scripts/publish_hf_model.py \
  --weights /content/chessfly-flynet.safetensors \
  --source-dir /content/flychess \
  --prune-source-mirror
```

The publisher validates the tensor keys and shapes, checks the source path
hygiene, runs the test suite, records a SHA-256 receipt, uploads the model,
removes only the audited source-mirror paths, and downloads the published
weights again to verify the remote checksum. It never prints credentials or
the Hub account name.

Useful options:

```text
--engine PATH       Stockfish executable (default: stockfish on PATH)
--fly-color COLOR   white or black
--depth N           Stockfish search depth
--max-plies N       Stop after N half-moves
--seed N            Select a reproducible surrogate policy
--connectome PATH   Use a validated connectome edge-list policy
--neural-steps N    Connectome simulation steps per fly turn
--fen FEN           Start from a specific position
--record PATH       Write a replayable JSONL experiment log
--result PATH       Write the final result JSON alongside --record
```

## Development roadmap

1. Fit the exposed dynamics parameters against held-out functional targets and
   report calibration error separately from chess performance.
2. Align MaleCNS annotations with FlyWire cell types and morphology without
   merging the male and female raw connectomes.
3. Replace the generic graph's handcrafted sensory projection with a documented
   photoreceptor interface.
4. Replace the generic output hash with a calibrated, experiment-specific
   neural readout while keeping legal-move masking outside the brain model.
5. Add reward-conditioned experiments and frozen/shuffled controls before
   making any claim about learning or chess skill.

## Scientific boundary

This project will use precise language: a simulated connectome is not a living
fly, and a game-playing demo is not evidence of consciousness or biological
equivalence. Any future connectome backend will document which parts come from
measured data and which parts are engineering assumptions.
