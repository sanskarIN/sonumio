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
