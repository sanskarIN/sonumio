# Studio instrument auditions

The studio mixer exposes an audition button for every instrument. The dry
mode uses the same shared attenuation as the verified eight-stem ZIP archive:
the selected WAV equals its corresponding archive member and retains the
original relative instrument levels. It ignores the static mix and room.

Solo mode temporarily selects only the requested track, overriding current
mute/solo switches without modifying the actual mixer. It preserves gain,
score-clock automation, room response and peak ceiling. Solo WAV files are
individually mastered previews; they must not be summed as raw stems.

The plan endpoint requires a score digest. Both audition modes validate the
supplied mixer and automation settings before admission to the existing
bounded single-worker music queue. A dry result identifies its common source
peak and integer PCM receipt; a solo result identifies the original and
effective automation digests. Neither mode changes the original score.

The browser validates the plan against the selected score and track, checks
the result against the plan, and independently checks WAV length, canonical
stereo PCM16 geometry and SHA256 before downloading. Playback uses the local
temporary audio URL. A recovered journal reference may not retain the full
audition receipt and is not described as a newly verified download.
