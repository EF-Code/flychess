# Connectome research for flychess

Research and architecture note, 2026-09-11. This document covers the
authoritative MaleCNS and FlyWire releases, their machine-readable formats and
terms, the limits of simulation claims, and an adapter contract for the
flychess experiment. It is deliberately a design note, not a claim that a
connectome is a complete biological brain simulation.

Throughout this document:

- **Evidence** means a fact stated by, or directly observable in, a linked
  primary source.
- **Engineering assumption** means a proposed flychess convention that is not
  supplied by the connectome release.
- **Open** means that the release does not provide enough information to make a
  biological or licensing claim without further validation.

## Executive recommendation

Use MaleCNS v1.0 as the default structural source for a future backend. The
official [MaleCNS download page](https://male-cns.janelia.org/download/) is the
release manifest for the flat connectome exports, and the
[MaleCNS release page](https://male-cns.janelia.org/release/) identifies v1.0
as the June 2026 release. Pin the exact release, artifact URL, retrieval time,
hash, filters, and node-selection rule in a manifest; do not silently consume
the latest live database.

Start with a reduced, annotated sensorimotor subgraph and an explicit sparse
runtime. A full graph is useful for offline analysis, but it is not evidence
that a real-time, biophysically faithful fly brain can be run on a chess
position. The first useful milestone is a reproducible structural experiment:
fixed visual stimuli, fixed graph snapshot, frozen readout, and external chess
legality.

Treat FlyWire v783 as a comparative/reference dataset, not as a drop-in
replacement for MaleCNS. It is a female adult brain release with separate
root-ID/versioning and a more restrictive public-release license. A FlyWire
backend should be an explicit configuration with its own provenance and
licensing checks.

## 1. What the releases actually represent

### MaleCNS

**Evidence.** The official MaleCNS site describes a whole male central nervous
system release covering the central brain, optic lobes, and ventral nerve cord.
The accompanying paper/preprint reports 166,691 neurons, 11,691 cell types, and
a structural graph with 25.6 million directed graph edges between 166,391
neurons. It also reports automated synapse detection quality and incomplete
proofreading/coverage; these are measurements of the released reconstruction,
not guarantees of biological completeness. See the
[MaleCNS paper preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)
and the current [MaleCNS v1.0 download manifest](https://male-cns.janelia.org/download/).

**Engineering implication.** A MaleCNS edge is a structural connection or
synapse-count aggregate. It is not automatically a conductance, a firing
probability, a delay, or a learned policy. The runtime must make those
translations configurable and visible in its provenance.

### FlyWire

**Evidence.** The public FlyWire release used by the published connectivity
dataset is materialization 783, a female adult brain snapshot. The release
contains segmentation/root IDs, synapse detections, predicted
neurotransmitter probabilities, and proofread connectivity products. FlyWire
also states that segmentation, proofreading, annotations, and synapse models
have different provenance and that the public data continue to be improved.
See [FlyWire's data provenance page](https://codex.flywire.ai/about_flywire),
the [FlyWire v783 Zenodo record](https://zenodo.org/records/10676866), and the
[public-release guidelines](https://home.flywire.ai/guidelines).

**Engineering implication.** MaleCNS body IDs and FlyWire root IDs are
different identifier systems. Never join them because their numeric values
happen to look comparable. Use an explicit cross-match/type table and record
the matching source.

### Count definitions must be pinned

**Evidence.** The MaleCNS paper reports a raw graph count, while the current
[FlyWire Codex download/API page](https://codex.flywire.ai/api/download) shows
summary counts using its own release/query definition. The public
[MaleCNS supplemental analysis notebook](https://github.com/flyconnectome/2025malecns/blob/main/supplemental_data/quantify-neuron-connections.ipynb)
(which is labelled for the historical v0.9 release) applies a weight threshold
in one analysis.

**Engineering rule.** Do not use a dashboard count as the adjacency cardinality.
Every experiment should record: release, input artifact, row/edge semantics,
minimum-weight/confidence filters, node filters, and the resulting node/edge
counts. Counts that cannot be reconciled are an operational warning, not a
reason to silently substitute one dataset for another.

## 2. Authoritative formats and concrete fields

The current MaleCNS bulk artifacts are under the
[official v1.0 flat-connectome bucket](https://male-cns.janelia.org/download/).
The links below are examples of the release's named artifacts; a loader should
resolve them from a pinned manifest rather than construct “latest” paths.

### MaleCNS flat exports

| Artifact | Documented fields/semantics | Units and cautions |
| --- | --- | --- |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | Directed segment/body aggregate. The official supplemental analysis reads **body_pre**, **body_post**, and **weight**; the download page describes this as segment-to-segment connection strength. | **weight** is a synapse-count-like structural strength, not a calibrated synaptic conductance. |
| syn-partners-male-cns-v1.0-minconf-0.5.feather | One row per pre/post partner pair. The published schema names **x_pre**, **y_pre**, **z_pre**, **body_pre**, **conf_pre**, **x_post**, **y_post**, **z_post**, **body_post**, **conf_post**, and **primary_post**. | Coordinates are in 8 nm voxels. **primary_post** is the primary neuropil label. |
| syn-points-male-cns-v1.0-minconf-0.5.feather | Synapse-point table with body/segment IDs and ROI information; pre- and post-synaptic points are separate rows identified by **kind** such as **PreSyn** and **PostSyn**. | A single presynaptic site can have multiple postsynaptic partners. Do not infer one-to-one synapses from point rows. |
| tbar-neurotransmitters-male-cns-v1.0.feather | Per-presynaptic-site neurotransmitter prediction probabilities. | Probabilities are predictions, not receptor-level effects or signed weights. |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | Curated body-level classes, types, sides, and related annotations. The release page does not promise an exhaustive stable column list for every future export. | Discover and record the actual Arrow schema at ingestion. Do not hard-code a column solely from a display label. |
| body-stats-male-cns-v1.0-minconf-0.5.feather | Body-level summary statistics, including synapse-count summaries. | Summary statistics are not a replacement for partner-level edges. |

The official [MaleCNS supplemental repository](https://github.com/flyconnectome/2025malecns)
is useful for derived sensorimotor flow, traversal, optic-column, and
cross-match products. Those products are analysis outputs, not the primary
connectome. In particular, sensorimotor flow tables can prioritize candidate
routes; they do not establish that a visual neuron pool means “the e4 square”
or that a motor pool means “move the knight.”

### MaleCNS annotations and neuPrint-shaped data

**Evidence.** The MaleCNS paper describes annotations including
**superclass**, **hemilineage**, **somaSide**, **somaNeuromere**, **entryNerve**,
**exitNerve**, **class**, **subclass**, synonyms, and parallel MaleCNS/FlyWire/
hemibrain type matches. The [neuPrint user guide](https://neuprint.janelia.org/public/neuprintuserguide.pdf)
documents the query-facing neuron fields **bodyId**, **cropped**, **instance**,
**post**, **pre**, **cellBodyFiber**, **roiInfo**, **size**, **somaLocation**,
**somaRadius**, **status**, **statusLabel**, **timeStamp**, and **type**, plus ROI
flags. Its chemical-synapse model includes **Synapse.confidence**,
**location**, and **type** (**pre** or **post**), with **ConnectsTo.weight** and
**weightHP** on relationships. The [neuprint-python query documentation](https://connectome-neuprint.github.io/neuprint-python/docs/queries.html)
shows the common adjacency result fields **bodyId_pre**, **bodyId_post**,
**roi**, and **weight**.

**Engineering contract.** Normalize source-specific names to an internal
directed edge record:

~~~text
pre_id, post_id, weight, roi, confidence, source_row_id
~~~

Keep **pre_id** and **post_id** as source identifiers until an internal
contiguous index is built. Keep the source IDs in every trace.

### Spatial and skeleton formats

**Evidence.** MaleCNS aligned EM and segmentation volumes are published at
8 nm isotropic resolution, with N5/precomputed access patterns. MaleCNS also
publishes SWC skeletons in EM coordinates and a 1 nm precomputed skeleton
variant. The official download page gives the exact volume and skeleton
locations.

For interoperable spatial data, the [Neuroglancer precomputed skeleton
specification](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/skeletons.md)
defines an **info** JSON object with **@type: neuroglancer_skeletons**, a
12-number 4-by-3 transform, vertex attributes, and binary segment files
containing little-endian vertex/edge counts, float32 vertex positions, and
uint32 edge indices. The [precomputed datasource documentation](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed)
defines the **precomputed://** URL convention.

**Engineering implication.** Chess does not require raw EM volumes or
morphology for the first backend. Keep those as optional provenance/context
inputs; use the flat graph and annotations for the initial runtime.

### FlyWire Feather and CAVE formats

**Evidence.** The official [FlyWire v783 connectivity record](https://zenodo.org/records/10676866)
documents **flywire_synapses_783.feather** with:

- **id**;
- **pre_pt_root_id**, **post_pt_root_id**;
- **connection_score** and **cleft_score**;
- predicted NT probability columns **gaba**, **ach**, **glut**, **oct**,
  **ser**, **da**;
- **neuropil**;
- **pre_pt_position_{x,y,z}** and **post_pt_position_{x,y,z}** in nanometers.

Its proofread connection table contains **pre_pt_root_id**,
**post_pt_root_id**, **neuropil**, **syn_count**, and average predicted-NT
columns such as **gaba_avg**, **ach_avg**, **glut_avg**, **oct_avg**,
**ser_avg**, and **da_avg**. The full synapse Feather is large; the record
documents chunked Arrow reads.

For live/materialized access, [CAVEclient's materialization guide](https://www.caveconnecto.me/CAVEclient/tutorials/materialization/)
documents versioned annotation tables, exact materialization versions,
segmentation lineage, **desired_resolution**, and filtered synapse queries.
The [schema guide](https://www.caveconnecto.me/CAVEclient/tutorials/schemas/)
documents the JSON Schema service. CAVE queries can be large and are bounded;
the guide documents a 200,000-row query limit and warns that unpinned “latest”
queries are not consistent across a long analysis.

**Engineering rule.** For FlyWire, store both the materialization version or
timestamp and the segmentation/root-ID lineage metadata. For a reproducible
bulk experiment, prefer the versioned Zenodo Feather over an unpinned live
query.

## 3. Licensing and provenance

| Source | Evidence from the source | Flychess consequence |
| --- | --- | --- |
| MaleCNS | The official MaleCNS pages state that Male CNS data are CC BY; the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) permits sharing/adaptation with attribution, a license link, and change indication, subject to other rights. | Carry attribution and source/version links into every derived artifact. Do not imply that a derived simulator is an official MaleCNS product. |
| FlyWire public release | FlyWire's [public-release guidelines](https://home.flywire.ai/guidelines) state CC BY-NC 4.0 for the public release. The [community principles](https://edit.flywire.ai/principles.html) add publication/contributor expectations, especially for prepublication or edited material. | Treat FlyWire-derived data as non-commercial by default, preserve required citations, and review the terms before publishing a cache, hosted demo, or commercial use. |
| CAVEclient software | The [CAVEclient repository](https://github.com/CAVEconnectome/CAVEclient) identifies the client code as MIT-licensed. | MIT licensing of the client does not relicense the FlyWire data returned by the client. |

**Engineering policy.** Do not commit raw Feather, CAVE dumps, volume
cutouts, or derived adjacency caches to this repository by default. The
repository should contain a loader, a manifest template, and documentation;
users should fetch the upstream data under the upstream terms. If a derived
cache is later distributed, it must carry:

~~~yaml
source: malecns | flywire
release: v1.0 | 783
materialization: null
source_urls: []
retrieved_at_utc: null
sha256: {}
filters: {}
node_selection: {}
license_notice: ""
changes: ""
~~~

This is an engineering policy, not legal advice. The exact upstream terms and
the user's intended distribution context still control.

## 4. Realistic simulation limits

### Evidence-based limits

1. **A connectome is structural.** The MaleCNS paper explicitly presents the
   reconstruction as a wiring diagram for circuit analysis; functional
   interpretation and validation remain necessary.
2. **Coverage and proofing are imperfect.** The paper reports automated
   detection metrics and partial pre/post-synaptic completion. The public
   reconstruction should be treated as a measured, error-prone snapshot.
3. **The graph does not provide a complete neuron model.** The releases do not
   supply, for every neuron and synapse, calibrated membrane capacitance, leak,
   ion-channel dynamics, receptor conductance, synaptic time constants,
   transmission delay, gap-junction behavior, neuromodulatory state, or
   plasticity rules.
4. **Neurotransmitter labels are not synaptic signs.** MaleCNS and FlyWire
   publish predictions/probabilities and aggregate labels. A label does not
   determine receptor expression or whether a downstream effect is excitatory,
   inhibitory, or state-dependent.
5. **Chemical-only data are incomplete for electrical behavior.** The FlyWire
   paper describes chemical synapses and treats electrical connections as a
   future/other modality; absence from the table is not evidence of absence in
   the animal.
6. **Information flow is not latency.** Connectome flow analyses and “early” or
   “late” labels do not by themselves supply a biophysical timing model.
7. **Causal response is state-dependent.** The [FlyWire effectome paper](https://www.nature.com/articles/s41586-024-07982-0)
   uses a sparse structural graph as a prior and stresses that causal effects
   are not identical to connectivity. Its signed weights, thresholds, and
   linear recurrent model are a particular analysis model, not a universal
   connectome simulator.

### Engineering limits and assumptions

- A full graph at this scale should use sparse, memory-mapped or chunked
  structures. A dense neuron-by-neuron matrix is not a reasonable baseline.
- The first runtime should be a reduced graph selected by explicit annotation,
  ROI, and edge filters. “Real time” must be a benchmark result, not a design
  promise.
- The initial biological model should be labelled, for example,
  **rate**, **leaky_integrator**, or **lif**; its **dt_ms**, state
  initialization, thresholds, decay, delays, and weight normalization are
  experiment parameters.
- Provide at least three synapse policies: **unsigned_count**, a documented
  signed/NT scenario, and **calibrated**. The signed mapping must be a
  scenario with sensitivity tests, not a hidden assumption copied from one
  paper.
- Keep an explicit null/control family: shuffled edges with degree preserved,
  random readout, frozen sensory drive, edge-threshold sweeps, and a symbolic
  baseline. Without these controls, a chess score cannot establish that
  connectome structure caused the behavior.
- Do not describe the system as conscious, intelligent, or biologically
  equivalent to a fly. At this stage it is a connectome-informed controller
  with a chess adapter.

## 5. Recommended chess sensory/action adapter

### Boundary

The chess environment owns FEN state, legal move generation, turn order,
promotion rules, terminal outcomes, and board mutation. **python-chess**
remains the legality authority. Stockfish remains an opponent/evaluator
outside the connectome runtime. The brain receives a rendered stimulus and
produces activity/readout values; it does not receive Stockfish's evaluation or
a precomputed legal-move answer.

The current flychess surrogate accepts a symbolic board encoding and can be
given a legal-move list. That is a useful deterministic baseline, but it
should be reported as a rule-assisted symbolic policy. A connectome experiment
should return a distribution over candidate actions and apply legality after
the neural readout.

### Stable scene contract

BoardScene is an engineering interface:

~~~yaml
game_id: string
ply: integer
fen: string                 # environment trace; not injected into the brain
orientation: white_perspective | black_perspective
image:
  dtype: float32
  shape: [H, W, C]
  channel_semantics: []
  value_range: [0.0, 1.0]
stimulus_hash: sha256
side_to_move_cue: rendered | absent
renderer_version: string
~~~

Use a fixed, deterministic egocentric visual frame. Piece color/type can be
represented by stable color/shape/luminance channels; coordinates, algebraic
labels, FEN text, and Stockfish values should not be hidden in the image. If
side-to-move or check status is shown, render it as an explicit visual cue and
record that choice. A symbolic renderer may remain as a separate control.

### Vision adapter

VisionAdapter converts BoardScene.image into injections into an explicitly
versioned set of sensory/optic-lobe nodes:

~~~yaml
InputInjection:
  node_id: source_body_id_or_root_id
  t_ms: number
  amplitude: number
  polarity: number
  source_channel: string
  units: normalized_drive
~~~

Candidate pools may be selected using MaleCNS type/superclass/ROI annotations,
optic-column assignments, and the supplemental sensorimotor-flow products.
Those sources can identify plausible routes. They do not publish a biological
mapping from an 8-by-8 chess image to a meaningful retinal code. The mapping,
normalization, temporal filtering, and selected nodes are therefore
engineering assumptions that must be versioned and tested against controls.

### Sparse runtime contract

The runtime consumes a pinned graph snapshot and produces a traceable activity
frame:

~~~yaml
ConnectomeManifest:
  source: malecns | flywire
  release: string
  materialization_version_or_timestamp: string | null
  graph_artifact: string
  annotation_artifacts: []
  source_urls: []
  retrieved_at_utc: string
  sha256: {}
  node_id_namespace: body_id | root_id
  edge_semantics: directed_pre_to_post
  edge_weight_semantics: synapse_count_proxy
  edge_filter: {}
  node_selection: {}
  spatial_units: 8nm_voxel | nm | unknown
  neuron_model: rate | leaky_integrator | lif
  dt_ms: number
  synapse_policy: unsigned_count | signed_scenario | calibrated
  decoder_version: string
~~~

~~~text
step(manifest, state, input_injections, dt_ms, steps)
  -> ActivityFrame(selected_node_activity, summary_metrics, trace_id)
~~~

The runtime should map source IDs to contiguous internal indices, but expose
only the original IDs in logs and manifests. It should be deterministic for a
fixed seed and snapshot, support a bounded node readout, and make state,
initialization, normalization, edge filtering, and sign policy inspectable.
Learning/plasticity should be a separate experimental mode because the
connectome release does not define a chess reward circuit or a plasticity rule.

### Action decoder and legality gate

The decoder is a project-defined readout over selected downstream/descending
or motor-related pools:

~~~yaml
ActionDistribution:
  from_square_logits: [64]
  to_square_logits: [64]
  promotion_logits: [none, queen, rook, bishop, knight]
  abstain_logit: number
  decoder_version: string
  output_node_ids: []
~~~

There is no evidence in the releases for a neuron that naturally means a
particular chess origin square, destination square, or promotion piece.
Mapping activity to those fields is an engineering readout and should be
trained/evaluated independently of the graph snapshot.

The environment then performs the legality gate:

~~~text
candidate_moves = decode_all_uci_moves(action_distribution)
legal_moves = python_chess.Board.legal_moves
masked_moves = candidate_moves intersect legal_moves
choose(masked_moves, fallback_or_abstain_policy)
~~~

This gate is an interface invariant, not a biological claim. Record whether a
run used a legal mask, a fallback, or abstained. If a future experiment feeds
the legal mask into the neural runtime, label it explicitly as a
rule-assisted condition.

### Reward and reproducibility contract

Use terminal outcome as the primary reward event; optional evaluation shaping
must be labelled separately:

~~~yaml
RewardEvent:
  game_id: string
  ply: integer
  outcome: -1 | 0 | 1
  shaping_delta_cp: number | null
  source: game_result | stockfish_shaping
~~~

For every action trace, retain the pre-move FEN, renderer/version and stimulus
hash, graph manifest/hash, injections, selected activity, decoder output,
legal-candidate/mask hash, chosen UCI move, post-move FEN, and reward event.
This makes it possible to distinguish a graph effect from a renderer,
decoder, legality, or Stockfish artifact.

## 6. Staged implementation

1. **Control:** keep the existing deterministic symbolic surrogate and
   Stockfish game loop as the reproducible baseline.
2. **Stimulus:** add a versioned fixed board renderer and test it with
   synthetic board fixtures; do not claim biological vision yet.
3. **Small graph:** select a documented visual-to-association-to-output
   subgraph, use an unsigned sparse rate/leaky model, and freeze the decoder.
4. **Ablations:** sweep edge thresholds and sign policies; compare degree-
   preserved shuffles, random readouts, and symbolic controls.
5. **Calibration:** compare intermediate activity/readouts with published fly
   visual or motor observations before making a biological performance claim.
6. **Scale:** only then benchmark a larger MaleCNS snapshot and decide whether
   an offline full-graph runner is useful.

The success criterion for the first connectome milestone should be
reproducibility and a clearly defined action interface, not an assertion that
the fly has learned chess.

## Primary sources

- [MaleCNS official download and format manifest](https://male-cns.janelia.org/download/)
- [MaleCNS release notes](https://male-cns.janelia.org/release/)
- [MaleCNS paper/preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)
- [MaleCNS supplemental/derived products](https://github.com/flyconnectome/2025malecns)
- [neuPrint user guide](https://neuprint.janelia.org/public/neuprintuserguide.pdf)
- [FlyWire provenance and data sources](https://codex.flywire.ai/about_flywire)
- [FlyWire public-release guidelines](https://home.flywire.ai/guidelines)
- [FlyWire community principles](https://edit.flywire.ai/principles.html)
- [FlyWire v783 connectivity dataset](https://zenodo.org/records/10676866)
- [CAVEclient repository](https://github.com/CAVEconnectome/CAVEclient)
- [CAVE materialization/versioning guide](https://www.caveconnecto.me/CAVEclient/tutorials/materialization/)
- [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed)
- [FlyWire effectome paper](https://www.nature.com/articles/s41586-024-07982-0)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [FlyWire Terms of Service](https://flywire.ai/tos)
