# Legacy comparison artifact runtime

> This document describes a legacy compatibility path used only for controlled
> comparison. It is not FlyNet provenance, it is not used by the from-scratch
> training pipeline, and its weights must never be presented as Flychess-trained
> weights. The primary Flychess model release is documented in
> `docs/huggingface-model-card.md`.

Flychess has an optional compatibility runtime for a public external model and
its companion demo. Use this path only for interoperability and controlled
comparative runtime tests. It is not part of FlyNet training and is not a
source of FlyNet weights or biological calibration data.

## Artifact boundary

The model repository publishes `flynet.safetensors` and metadata, but not the
connectome graph. The demo Space publishes `data/connectome.bin.gz` and
`data/neurons.bin.gz`. Acquire them separately and record SHA-256 checksums in
the experiment manifest. FlyWire's applicable non-commercial terms and the
source papers remain the authority for graph-derived artifacts.

The loader validates the binary sizes against the graph header, validates
neuron group labels, and derives the input and readout ports from the public
group convention:

| group | meaning | runtime role |
| --- | --- | --- |
| 0 | other | readout |
| 1 | optic | not directly read out |
| 2 | input | encoder destinations |
| 3 | descending | readout |

The public binary is CSR by presynaptic source. `ChessFlyGraph` performs the
same stable transpose as the demo worker so each postsynaptic row computes
`sum(W[target, source] * h[source])`. Edge signs are preserved as sign-only
values and the learned `log_gain` tensor supplies `exp(log_gain)` magnitudes.

## Forward contract

`ChessFlyModel.forward` accepts a batch of 780-feature canonical positions and
returns policy logits for 1,968 UCI actions plus 64 value-bin logits. It uses:

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

Black-to-move positions are mirrored into a white perspective before
encoding, then candidate UCI moves are mirrored back after the policy scores
them. The legal-move mask is applied by `ChessFlyPolicy`, outside the neural
graph, and every selected move is exposed through the existing
`DecisionReadout` telemetry interface.

## Biological boundary

This adapter reproduces a published engineering artifact's computation. It
does not turn the model into a calibrated biological MaleCNS brain, establish
that the learned decoder has natural motor semantics, or provide evidence of
consciousness. MaleCNS and FlyWire remain separate source families; future
calibration work must carry source identity, node-ID namespace, graph version,
sign policy, and held-out functional targets explicitly.
