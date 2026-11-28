# Native project arrangement assembly

The score assembly recipe can now be pinned to complete native projects.
This preserves compatible gain/balance automation along the same bar
spans, including repeats and reordered segments.

    python -m sonumio.music_project_assembly_cli pin score-recipe.json --source a=source.sonumio --title "New arrangement" --output project-recipe.json
    python -m sonumio.music_project_assembly_cli inspect project-recipe.json --source a=source.sonumio
    python -m sonumio.music_project_assembly_cli assemble project-recipe.json --source a=source.sonumio --output assembled.sonumio

The pin command accepts a validated score assembly recipe and verifies
the complete result before publishing the project recipe. Sources must
match the score digests in that recipe and use identical static mixers.
Project identity pins also bind automation and revision metadata.

Automation transfer is conservative. Missing gain lanes are unity and
missing balance lanes are zero. Step transitions may jump at a seam;
a nonconstant continuous ramp must meet the next span's initial value.
A partial nonconstant smoothstep is rejected because the original cubic
cannot generally be represented by a shortened smoothstep. Nonconstant
ramp sample durations must be preserved by clock translation, including
integer sample rounding. Linear crops preserve endpoint values but do
not claim bit-identical floating-point interpolation. The 64-key limit
is enforced after removal of redundant constant step keys.

The result is a new root project, not a revision of any one source.
No audio is rendered. The receipt records all source project digests
and the resulting score, automation and project identities.
