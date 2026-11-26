# Original symbolic melody model research

This is a narrow learned melody experiment. It predicts relative pitch
tokens from previous pitch plus supplied style, mode, chord degree,
within-bar onset and section. The procedural arrangement supplies
rhythm, harmony and accompaniment. It is not unrestricted text-to-music,
a waveform generator, or a released Sonumio-1 model.

## Corpus and model

Generate 8-bar original synthetic arrangements at a fixed key from
12 training seeds and 3 disjoint development seeds, each spanning
three styles and two modes. Convert the lead into 25 relative pitch
tokens, with explicit BOS and padding inputs. Teacher-forced targets
are shifted by one, and padded targets are ignored in loss. The source
receipt records split digests, full sequence deduplication and removal
of development sequences matching training pitch sequences exactly.
The remaining development panel must retain all style/mode strata.
This does not exclude near-duplicate phrases or establish real-world
generalization.

A small GRU sums context embeddings with the previous-pitch embedding,
normalizes, runs the recurrent state and projects to pitch logits.
The experiment uses CPU float32 AdamW, 128 updates, batch size 8, and
three independent initialization seeds. The uniform, smoothed unigram
and smoothed hierarchical conditional bigram controls are trained only
on the training panel. Compare *unmasked* teacher-forced negative
log likelihood, perplexity, top-1 accuracy and allowed probability
mass. The conditional count model falls back from exact contextual
bigram to context-only, previous-pitch and global counts.

## Free-running generation

At evaluation, use only context and the generated previous pitch,
never the held-out target pitches. For one development schedule per
style/mode, sample two seeds with and without constraints. The constrained
arm restricts pitches to the scale, chord tones at strong onsets and
tonic at the last note. These are hard rules, not learned abilities.
Measure unmasked validity, exact sequence matches, interval movement and
pitch variety. Replace the original lead with a generated constrained
melody and render one actual stereo WAV in memory per model seed; report
signal metrics and byte geometry, but do not publish audio or weights.

```bash
python -m research.music_melody_study --model-seeds 17,29,41 \
  --training-seeds 0,1,2,3,4,5,6,7,8,9,10,11 \
  --development-seeds 101,102,103 --updates 128 --batch-size 8 \
  --width 48 --bars 8 --sample-seeds 701,709
```

A useful first result is improvement over the untrained model and the
conditional bigram on every seed. Failure is a research outcome, not a
test crash. A better likelihood is not proof of pleasant composition,
long-form structure, novelty or copyright clearance. Independent test
scores, genuinely diverse rights-reviewed music, rhythm/harmony
generation and listening studies remain future work. No external
music or model weights are required or written.


## Verified first training result

The [dedicated CPU workflow](https://github.com/sanskarIN/sonumio/actions/runs/45841889713)
at commit 0e19825f719d6656d40c31037ce88ab947efffe1 passed 154
music tests and the three-seed study. It used 72 unique training
arrangements and 18 development arrangements (624 held-out tokens).
No exact relative pitch sequence crossed the split.

| Model | Mean held-out NLL |
|---|---:|
| Uniform | 3.218876 |
| Unigram | 2.771155 |
| Conditional bigram | 2.251636 |
| GRU seed 17 | 1.558139 |
| GRU seed 29 | 1.602278 |
| GRU seed 41 | 1.517601 |

Each GRU has 19,177 parameters. Mean trained NLL is 1.559339.
Unmasked free-running token validity is 0.85644, 0.84823 and 0.89381.
Masked validity is 1 by construction. None of the generated sequences
matched an entire training relative pitch sequence exactly; this does
not rule out phrase reuse or establish copyright clearance.
The learned melody is still a synthetic pilot.

## Joint bar structure model

A separate GRU factors chord/rhythm prediction into a causal chord
head and a rhythm head conditioned on that bar's chord. It uses seven
diatonic chord degrees and eight explicit lead-onset patterns. Context
is style, mode, section, phrase position and final-bar indicator.
Teacher-forced likelihood is unmasked; free-running sampling uses only
its own history and optional cadence/nonempty-finale constraints.

The source generator has finite structural variety, so exact structural
overlap is expected despite disjoint source seeds. The report separates
all development rows from novel full token sequences and reports null
metrics for an empty novel subset. Rebuild pitched backing and lead
rhythm from sampled tokens, with procedural pitches and instrument
rendering; no target lead or harmony is reused. The chord/rhythm model
does not yet jointly train with the melody model.

```bash
python -m research.music_structure_study
```

Uniform, unigram, context and conditional counts are train-only
comparators. CI runs three seeds, 128 updates each, and an actual
in-memory audio render per seed. The new model's performance is not
established before running the experiment.
