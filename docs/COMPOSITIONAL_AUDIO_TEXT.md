# Sonumio compositional sound-category controls

**Experimental first-party synthetic audio–text learning. No model weights, WAVs, release tags, pretrained embeddings or perceptual-quality claim.**

## The semantic-negative problem

The project's 21 synthetic sound categories consist of six atomic
sound types and all fifteen distinct two-source mixtures.

Some training descriptions are related but not identical:

- `rain` vs `rain`: **exact same category**.
- `rain` vs `rain and wind`: **partially shared** source components.
- `rain and wind` vs `wind and ocean`: **partially shared**.
- `rain and wind` vs `forest and ocean`: **disjoint** source components.

A simple one-positive-per-row InfoNCE contrastive loss pushes every
nonmatching category away. This can mistakenly treat a description
containing **some of the same sound sources** as a hard negative,
although partial overlap is legitimately meaningful. A partially
related description is neither an exact positive nor completely
unrelated; the distinction should be explicit rather than hidden in
the optimizer.

`research/compositional_alignment.py` adds a bounded category-graph
and differentiable related-category-masked loss for the original
Sonumio text and audio towers.

## Audited compositional category relationships

For a category's atomic source IDs (each has one or two IDs):

```text
exact             : source-id sets equal
partially_shared  : nonempty intersection, unequal sets
disjoint          : intersection empty
```

For all 21 categories, there are 441 ordered category pairs:
- 21 exact self-pairs;
- 180 partly shared ordered pairs;
- 240 fully disjoint ordered pairs.

These counts are exact and tested directly against the original
SYNTHETIC_COMBINATIONS definitions. The relationship matrix is
symmetric. The category data are **procedural DSP metadata**, not
crowdsourced semantic judgments about real-world sound descriptions.

## Related-category-masked symmetric InfoNCE

Let `A` and `T` be L2-normalized audio and text features. For
a matched row, define permitted comparison columns as:

```text
{its one exact positive} UNION {source categories disjoint from it}
```

Then the matched paired InfoNCE objective is:

```text
logits = A @ T.T / temperature
masked_logits[i, j] = logits[i, j]  when exact(i,j) or disjoint(i,j)
masked_logits[i, j] = -inf          when partially_shared(i,j)
L = (CE(masked_logits, diagonal)
     + CE(masked_logits.T, diagonal)) / 2
```

Each category must have at least one **truly disjoint** negative.
Both loss directions share the same category graph, and every
partial-overlap nonpositive pair is excluded. Duplicated exact
category identities are rejected rather than silently turned into
false negatives or extra positives. Malformed names, nonfinite
embeddings, wrong dtype/device and invalid temperature fail closed.

This design does NOT automatically make partially matching captions
valid exact labels. It merely avoids actively pushing related sounds
apart. The loss is differentiable through Sonumio's original small
audio convolution tower and original UTF-8 text Transformer.

## Compositional retrieval diagnostic

`compositional_retrieval` scores the original normalized embedding
matrix by acoustic source relations. For each category it reports:

- exact diagonal similarity and full 21-way matching rank;
- mean similarity to partly shared source combinations;
- mean similarity to fully disjoint source combinations;
- exact-minus-disjoint, partial-minus-disjoint, exact-minus-partial
  margins, with explicit negative/partial candidate counts;
- grouped descriptive means/min/max and exact top-1 rates.

No artificial acoustic independence is inferred from related
source category combinations. Missing partial pairs are reported as
`null`, not silently replaced with a made-up zero similarity.

The bilingual evaluation accepts an original
`OriginalAudioTextAligner` (trained elsewhere or randomly
initialized) and first-party synthetic held-out source seeds.
It uses **one acoustic recording per source group**, with the same
waveform for English and Hindi text. The test runs both languages,
an evaluation-only wrong-category cyclic caption shuffle, and
same-category English/Hindi text-vector cosine.

The model and global RNG are checked before and after, no neural
state, WAV or embedding bytes are serialized. The result is numeric
metadata only. A randomly initialized network's scores cannot be
interpreted as useful text/audio alignment.

## Reproduce the deterministic controls

```bash
python -m pytest -q tests/test_compositional_alignment.py
```

```python
from research.compositional_alignment import (
    SOURCE_GROUPS, category_relation_matrix,
    overlap_masked_info_nce, evaluate_bilingual_composition
)

relations, audit = category_relation_matrix(SOURCE_GROUPS)
assert audit["ordered_partially_shared_relations"] == 180

# audio_features and text_features are the output of ORIGINAL
# trainable Sonumio encoders, with identical row/group ordering.
loss, report = overlap_masked_info_nce(
    audio_features, text_features, SOURCE_GROUPS,
    temperature=0.1,
)
loss.backward()
```

The dedicated PyTorch CI covers the exact 21-source relation graph,
mathematical positive/partial/disjoint controls, derivative gradients
through both original encoder towers, perfect and shuffled embedding
rankings, malformed/nonfinite inputs, and the actual bilingual
first-party held-out panel.

## What this cannot establish

Synthetic sound source overlap is not equal to true human semantic
relatedness. A mixture of rain and a bird call might not be an
interchangeable perceptual event with either individual source. The
current small first-party dataset uses templated captions and
correlated acoustic source families.

Before training a general Sonumio model using this loss, test actual
hard negative mining on rights-cleared real recordings, many natural
language paraphrases, source/speaker disjoint split policies,
independent model seeds, semantic relevance judged by people, and
audio listening quality. Avoid automatically turning partial
overlap into confident positive labels.

No Sonumio-1 model generation, trained model weights, public audio,
sound-quality certification or software pre-release is authorized
by this experiment.
