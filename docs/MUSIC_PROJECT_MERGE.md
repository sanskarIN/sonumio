# Reconcile native project revisions

Two direct revisions of the same verified base project can be compared
and reconciled without a server. The reconciliation uses the existing
conservative score merge, and independent automation lane and mixer
gain domains. Mute/solo is treated as one routing domain. A concurrent
automation lane edit against a different score clock is a conflict.

    python -m sonumio.music_project_merge_cli diff base.sonumio left.sonumio
    python -m sonumio.music_project_merge_cli preview base.sonumio left.sonumio right.sonumio
    python -m sonumio.music_project_merge_cli merge base.sonumio left.sonumio right.sonumio --output merged.sonumio > receipt.json
    python -m sonumio.music_project_merge_cli verify base.sonumio left.sonumio right.sonumio merged.sonumio receipt.json

Conflicts return exit code 3 and do not publish a project. Validation
and I/O errors return 2. A successful merge is a revision of the left
project, not a multi-parent commit: native project v1 supports only one
parent. A result already present in left does not create a new revision.
The receipt records both siblings and the common base, the domain choices,
and a deterministic hash. Retain all inputs to replay it. The receipt
does not authenticate people or prove historical authorship.

Input and output digest pins are supported. The output is atomically
published, and an input path cannot be overwritten.
