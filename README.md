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
- tests covering graph validation, replay tampering, HTTP boundaries, and engine integration.

The surrogate is deliberately not presented as a biological fly brain. A
connectome gives us a wiring diagram, but calibrated neural dynamics,
chemical signaling, sensory transduction, and motor semantics still need to
be supplied. The policy interface is designed so a MaleCNS/FlyWire backend can
replace the surrogate without changing the game or engine layers.

## Run it

Use the shared Python environment required by this workspace:

```bash
/home/hiro/.venv/bin/python -m pip install -e '.[dev]'
/home/hiro/.venv/bin/flychess --depth 3 --max-plies 40
```

If the console script is not on the environment path, use:

```bash
/home/hiro/.venv/bin/python -m flychess.cli --depth 3 --max-plies 40
```

Serve the browser UI locally:

```bash
/home/hiro/.venv/bin/python -m flychess.web
```

The server binds to `127.0.0.1` by default. Open the printed URL and use the
bounded game form to run an experiment.

Record a replayable game:

```bash
/home/hiro/.venv/bin/flychess --depth 3 --max-plies 40 \
  --record runs/game.jsonl --result runs/result.json
```

Run the graph backend with a connectome edge-list JSON file:

```bash
/home/hiro/.venv/bin/flychess --connectome examples/tiny-connectome.json \
  --neural-steps 2 --depth 2 --max-plies 24
```

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

1. Add a provenance-aware importer for MaleCNS/FlyWire releases.
2. Replace the generic graph's handcrafted sensory projection with a documented
   photoreceptor interface.
3. Replace the generic output hash with a calibrated, experiment-specific
   neural readout while keeping legal-move masking outside the brain model.
4. Add reward-conditioned experiments and frozen/shuffled controls before
   making any claim about learning or chess skill.

## Scientific boundary

This project will use precise language: a simulated connectome is not a living
fly, and a game-playing demo is not evidence of consciousness or biological
equivalence. Any future connectome backend will document which parts come from
measured data and which parts are engineering assumptions.
