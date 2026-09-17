# Biological calibration boundary

Flychess treats a connectome release as a versioned data source, not as a
drop-in chess brain. The first calibration slice is now implemented in
`flychess.calibration`:

1. `DatasetManifest` records source, version, sex, dataset ID, license, and
   the local artifacts used.
2. Every artifact can be SHA-256 locked and verified before ingestion.
3. MaleCNS-like annotation, connectivity, and neurotransmitter rows are
   normalized into a filtered `CanonicalConnectome`.
4. `SparseLIF` runs bounded synchronous dynamics over that graph with explicit
   synapse scaling, sign defaults, and optional per-neuron LIF parameters.
5. Feather ingestion is optional and requires `pyarrow`; the base package does
   not silently download or vendor the large public files.

The importer and simulator intentionally do not claim calibrated biological
dynamics. The next layer must fit the exposed unknown parameters against
held-out functional targets. Chess remains a downstream task readout, with
legal-move masking outside the biological core.

## Local usage

Create a manifest whose artifact paths are relative to one data root:

```python
from flychess.calibration import DatasetManifest, import_malecns

manifest = DatasetManifest.from_json("data/malecns-v1.0-manifest.json")
graph = import_malecns(
    manifest,
    root="data",
    body_ids=[12781, 556329],  # prefer a documented subgraph first
)
print(graph.summary())

from flychess.calibration import SparseLIF

dynamics = SparseLIF(graph, inputs=[12781], outputs=[556329])
dynamics.run(8, stimulus={12781: 1.0})
print(dynamics.readout())
```

For production ingestion, the manifest must contain hashes for the
`annotations`, `connectivity`, and optional `neurotransmitters` artifacts.
Keep the raw Feather files outside Git and record their release URLs in the
manifest metadata.

Install the optional reader explicitly when the artifacts are available:

```bash
~/.venv/bin/python -m pip install -e '.[dev,calibration]'
```
