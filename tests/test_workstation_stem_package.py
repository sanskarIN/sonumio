"""Deterministic single-publication stem archives and resource policies.

Tests deliberately use PCM fixtures and a structural mock for the engine;
the real Sonumio audio_core is separately exercised by CI integration tests.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
import numpy as np
from pathlib import Path
import sys
import types
import wave
import zipfile

import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.publish import PublicationOptions
from sonumio.midi_workstation.stem_package import (
    STEM_PROTOCOL, MAX_STEM_PACKAGE_BYTES, MAX_TOTAL_PCM_SAMPLES,
    StemPackagePlan, StemPatch, _canon, _seed, _verify_pcm,
    inspect_stem_pack, preflight_stems, publish_stem_pack, render_stem_pack,
)


def score(*, drums=False, muted=False, pan=64, event_count=1, end=480):
    source=Score.empty(end=end)
    alpha=ScoreTrack("alpha", "Alpha", 0, 0, 0, 0,
              tuple(sorted((ScoreNote(f"a{i}",0,120,60+i%12,velocity=70,pan=pan)
                    for i in range(event_count)), key=lambda n:(n.start,n.pitch,n.duration,n.id))),muted=muted)
    beta=ScoreTrack("beta", "Beta", 1, 0, 0, 0,
                   (ScoreNote("b0",240,120,55),))
    tracks=(alpha,beta)
    if drums:
        tracks+=(ScoreTrack("drums","Drums",9,0,0,0,
                (ScoreNote("kick",0,80,36),)),)
    return replace(source,tracks=tracks)


def plan(**kwargs):
    args=dict(patches=(StemPatch("alpha",gain=.05,release_frames=32),
                       StemPatch("beta",gain=.05,release_frames=64)),
              sample_rate=8000, channels=2, bits=16)
    args.update(kwargs)
    return StemPackagePlan(**args)


def pcm(frames=4064, channels=2, rate=8000, bits=16, amplitude=0):
    buf=BytesIO()
    with wave.open(buf,"wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(bits//8)
        wav.setframerate(rate)
        wav.writeframes(bytes([amplitude])*(frames*channels*(bits//8)))
    return buf.getvalue()


def fixture_pack(*, a=None, b=None, sample_rate=8000, frames=4064,
                 channels=2, bits=16, manifest_changes=None):
    members=[("stems/alpha.wav", a if a is not None else pcm(frames=frames,channels=channels,rate=sample_rate,bits=bits)),
             ("stems/beta.wav", b if b is not None else pcm(frames=frames,channels=channels,rate=sample_rate,bits=bits))]
    manifest={"protocol":STEM_PROTOCOL,"score_sha256":"a"*64,
              "sample_rate":sample_rate,"channels":channels,"frames":frames,
              "bits":bits,"skipped_percussion_notes":0,"dither_seed":None,
              "original_synthesis":True,
              "stems":[{"track_id":name.split("/")[-1].split(".")[0],
                       "member":name,"sha256":sha256(data).hexdigest(),
                       "notes":1,"patch":asdict(StemPatch(name.split("/")[-1].split(".")[0])),
                       "source_audio_peak":0.,"source_audio_rms":0.,
                       "pcm_bytes":len(data)} for name,data in members]}
    if manifest_changes:
        manifest_changes(manifest)
    return pack_zip(_canon(manifest),members)


def pack_zip(manifest,members):
    stream=BytesIO()
    with zipfile.ZipFile(stream,"w") as archive:
        for name,content in [("manifest.json",manifest),*members]:
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            info.create_system=3
            info.external_attr=0o100644<<16
            archive.writestr(info,content)
    return stream.getvalue()


@dataclass(frozen=True)
class FakeAudio:
    frames:int
    channels:int
    sample_rate:int

    @property
    def samples(self):
        return np.zeros((self.channels,self.frames),dtype=np.float32)


@dataclass(frozen=True)
class FakeReport:
    rendered_notes:int
    output_peak:float=.12
    output_rms:float=.05


@dataclass(frozen=True)
class FakeResult:
    audio:FakeAudio
    report:FakeReport


@pytest.fixture
def fake_engine(monkeypatch):
    audio=types.ModuleType("sonumio.audio_core")
    audio.__path__=[]
    dither=types.ModuleType("sonumio.audio_core.dither")
    wav=types.ModuleType("sonumio.audio_core.waveio")
    errors=types.ModuleType("sonumio.audio_core.errors")
    contracts=types.ModuleType("sonumio.audio_core.contracts")
    contracts.make_buffer=lambda data,rate:FakeAudio(data.shape[1],data.shape[0],rate)
    class AudioContractError(Exception):pass
    class AudioProcessingError(Exception):pass
    class AudioLimitError(Exception):pass
    class AudioIntegrityError(Exception):pass
    errors.AudioContractError=AudioContractError
    errors.AudioProcessingError=AudioProcessingError
    errors.AudioLimitError=AudioLimitError
    errors.AudioIntegrityError=AudioIntegrityError
    @dataclass(frozen=True)
    class QuantizationSettings:
        bits:int
        dither:bool
        seed:int|None
    dither.QuantizationSettings=QuantizationSettings
    calls=[]
    def fake_render(subset, options):
        duration=int(subset.duration_microseconds*options.sample_rate/1_000_000)
        result=FakeResult(FakeAudio(duration+options.release_frames,options.channels,options.sample_rate),
                          FakeReport(sum(len(t.notes) for t in subset.tracks)))
        calls.append((subset,options))
        return result
    def encode_wav_bytes(audio, *,settings):
        return pcm(audio.frames,audio.channels,audio.sample_rate,settings.bits)
    wav.encode_wav_bytes=encode_wav_bytes
    for pkg in [audio,dither,wav,errors,contracts]:
        monkeypatch.setitem(sys.modules,pkg.__name__,pkg)
    monkeypatch.setattr("sonumio.midi_workstation.wavetable_bridge.render_score_wavetable", fake_render)
    return calls


@pytest.mark.parametrize("channels,bits",[(1,8),(1,16),(1,24),(1,32),
                                         (2,8),(2,16),(2,24),(2,32)])
def test_pcm_header_contracts(channels,bits):
    wav=pcm(frames=12,channels=channels,bits=bits)
    _verify_pcm(wav,rate=8000,channels=channels,frames=12,bits=bits)
    with pytest.raises(MidiFormatError):
        _verify_pcm(wav,rate=8000,channels=3-channels,frames=12,bits=bits)


def test_sample_clock_preflight_requires_all_active_pitched_tracks():
    result=preflight_stems(score(),plan())
    assert result.frames==4064 and result.tail_frames==64
    assert result.stems==2 and result.selected_note_count==2
    assert result.source_score_sha256==score().sha256
    assert result.as_dict()["audio_rendered"] is False


def test_missing_or_unknown_patch_refused():
    for routes in [(StemPatch("alpha"),),(StemPatch("alpha"),StemPatch("orphan"))]:
        with pytest.raises(MidiWorkstationError):
            preflight_stems(score(),plan(patches=routes))


def test_percussion_requires_explicit_skip_permission():
    with pytest.raises(MidiWorkstationError,match="Percussion"):
        preflight_stems(score(drums=True),plan())
    checked=preflight_stems(score(drums=True),plan(skip_percussion=True))
    assert checked.skipped_percussion_notes==1


def test_muted_tracks_require_explicit_policy_and_same_patch_coverage():
    source=score(muted=True)
    with pytest.raises(MidiWorkstationError):
        preflight_stems(source,plan())
    checked=preflight_stems(source,plan(include_muted=True))
    assert checked.selected_note_count==2


def test_mono_unpanned_only():
    with pytest.raises(MidiWorkstationError,match="Mono"):
        preflight_stems(score(pan=0),plan(channels=1))
    assert preflight_stems(score(),plan(channels=1)).channels==1


@pytest.mark.parametrize("options",[
    {"patches":tuple()}, {"sample_rate":7999}, {"sample_rate":48001},
    {"channels":0}, {"channels":3}, {"bits":12}, {"bits":64},
    {"include_muted":1}, {"skip_percussion":None},
    {"dither_seed":-1}, {"dither_seed":2**32},
    {"maximum_samples":0}, {"maximum_samples":MAX_TOTAL_PCM_SAMPLES+1},
    {"patches":(StemPatch("beta"),StemPatch("alpha"))},
    {"patches":(StemPatch("alpha"),StemPatch("alpha"))},
])
def test_plan_validation_rejects_noncanonical_or_unsafe(options):
    with pytest.raises(MidiWorkstationError):
        plan(**options)


@pytest.mark.parametrize("kwargs",[
    {"track_id":"/bad"}, {"track_id":""}, {"waveform":"noise"},
    {"gain":-1}, {"gain":1.1}, {"attack_frames":-1},
    {"decay_frames":-10}, {"release_frames":-1},
    {"sustain_level":3}, {"table_size":240},
    {"max_harmonics":0}, {"table_size":256,"max_harmonics":128},
])
def test_patch_rejects_invalid_engine_inputs(kwargs):
    args={"track_id":"alpha"};args.update(kwargs)
    with pytest.raises(MidiWorkstationError):
        StemPatch(**args)


def test_sample_budget_rejects_before_renderer():
    with pytest.raises(MidiLimitError,match="sample budget"):
        preflight_stems(score(),plan(maximum_samples=100))


def test_too_many_notes_rejected_before_render():
    with pytest.raises(MidiLimitError):
        preflight_stems(score(event_count=2048),plan())


def test_fake_renderer_aligns_all_tracks_on_original_clock(fake_engine):
    source=score()
    data,receipt=render_stem_pack(source,plan())
    assert receipt["stems"]==2
    assert receipt["frames"]==4064
    assert receipt["source_score_sha256"]==source.sha256
    assert [x[0].tracks[0].id for x in fake_engine]==["alpha","beta"]
    assert [x[0].end for x in fake_engine]==[480,480]
    assert [x[1].release_frames for x in fake_engine]==[32,64]
    assert inspect_stem_pack(data)["package_sha256"]==sha256(data).hexdigest()


def test_render_is_bitwise_repeatable_and_has_no_file_side_effects(fake_engine,tmp_path):
    first,receipt=render_stem_pack(score(),plan(dither_seed=7))
    second,_=render_stem_pack(score(),plan(dither_seed=7))
    assert first==second
    assert sorted(tmp_path.iterdir())==[]
    assert _seed(7,"alpha")!=_seed(7,"beta")
    assert receipt["original_mathematical_synthesis"] is True


def test_integrity_check_fails_on_pinned_digest_mismatch():
    packed=fixture_pack()
    with pytest.raises(MidiIntegrityError):
        inspect_stem_pack(packed,expected_sha256="0"*64)


def test_corrupted_wav_member_rejected():
    packed=fixture_pack(a=b"garbage"*10)
    with pytest.raises(MidiFormatError):
        inspect_stem_pack(packed)


def test_declared_digest_mismatch_rejected():
    packed=fixture_pack(manifest_changes=lambda m:m["stems"][0].__setitem__("sha256","f"*64))
    with pytest.raises(MidiIntegrityError):
        inspect_stem_pack(packed)


@pytest.mark.parametrize("field,bad",[
    ("protocol","wrong"),("score_sha256","unknown"),
    ("channels",3),("frames",0),("sample_rate",7000),
    ("bits",12),("stems",[]),("original_synthesis",False),
    ("skipped_percussion_notes",-1),("dither_seed",-2),
])
def test_malformed_manifest_rejected(field,bad):
    with pytest.raises((MidiFormatError,MidiWorkstationError,MidiLimitError)):
        inspect_stem_pack(fixture_pack(manifest_changes=lambda m:m.__setitem__(field,bad)))


def test_manifest_does_not_trust_note_count_or_member_length():
    for key,value in [("notes",-1),("pcm_bytes",9000000),
                      ("member","../bad.wav"),("track_id","../")]:
        def change(m):m["stems"][0][key]=value
        with pytest.raises((MidiFormatError,MidiWorkstationError)):
            inspect_stem_pack(fixture_pack(manifest_changes=change))


def test_noncanonical_manifest_rejected():
    package=fixture_pack()
    with zipfile.ZipFile(BytesIO(package)) as z:
        data=z.read("manifest.json")
        members=[(x,z.read(x)) for x in z.namelist()[1:]]
    with pytest.raises(MidiFormatError,match="canonical"):
        inspect_stem_pack(pack_zip(json.dumps(json.loads(data),indent=4).encode(),members))


def test_unexpected_zip_member_is_rejected():
    data=fixture_pack()
    with zipfile.ZipFile(BytesIO(data)) as z:
        rows=[(x,z.read(x)) for x in z.namelist()]
    for name in ("../outside.wav", "/tmp/unsafe.wav", "stems/../bad.wav",
                 "manifest.json/../../other", "stems/not-a-wav.bin"):
        with pytest.raises(MidiFormatError):
            inspect_stem_pack(pack_zip(rows[0][1],rows[1:]+[(name,b"x")]))


def test_archive_is_independently_authenticated_by_member_hashes():
    data=fixture_pack()
    assert inspect_stem_pack(data)["stems"]==2
    with zipfile.ZipFile(BytesIO(data)) as z:
        manifest=z.read("manifest.json")
        members=[(x,z.read(x)) for x in z.namelist()[1:]]
    change=bytearray(members[1][1])
    change[-1] ^= 0x40
    change=bytes(change)
    with pytest.raises(MidiIntegrityError):
        inspect_stem_pack(pack_zip(manifest,[(members[0][0],members[0][1]),(members[1][0],change)]))


def test_atomic_publication_and_no_clobber(fake_engine,tmp_path):
    source=score()
    destination=tmp_path/"stems.zip"
    receipt=publish_stem_pack(source,plan(),destination)
    assert receipt["publication"]["output_sha256"]==sha256(destination.read_bytes()).hexdigest()
    original=destination.read_bytes()
    with pytest.raises(MidiWorkstationError,match="overwrite"):
        publish_stem_pack(source,plan(),destination)
    assert destination.read_bytes()==original
    assert not list(tmp_path.glob("*.tmp"))


def test_pinned_output_digest_refuses_write(fake_engine,tmp_path):
    path=tmp_path/"stem.zip"
    with pytest.raises(MidiIntegrityError):
        publish_stem_pack(score(),plan(),path,PublicationOptions(expected_sha256="f"*64))
    assert not path.exists()


def test_publication_blocks_input_output_alias(fake_engine,tmp_path):
    destination=tmp_path/"source.zip"
    destination.write_bytes(b"important")
    with pytest.raises(MidiFormatError):
        publish_stem_pack(score(),plan(),destination,
            PublicationOptions(overwrite=True),source_path=destination)
    assert destination.read_bytes()==b"important"


def test_protected_recipe_blocks_overwriting_itself(fake_engine,tmp_path):
    source=tmp_path/"patch-recipe.json"
    source.write_bytes(b"do-not-overwrite")
    with pytest.raises(MidiFormatError):
        publish_stem_pack(score(),plan(),source,
            PublicationOptions(overwrite=True),protected_paths=(source,))
    assert source.read_bytes()==b"do-not-overwrite"


def test_symlink_destination_fails_closed(fake_engine,tmp_path):
    source=tmp_path/"original.zip";source.write_bytes(b"important")
    output=tmp_path/"output.zip"
    try: output.symlink_to(source)
    except (NotImplementedError,OSError):pytest.skip("Platform cannot create symlinks")
    with pytest.raises(MidiFormatError):
        publish_stem_pack(score(),plan(),output,PublicationOptions(overwrite=True))
    assert source.read_bytes()==b"important"


def test_output_extension_must_be_explicit_zip(fake_engine,tmp_path):
    with pytest.raises(MidiFormatError):
        publish_stem_pack(score(),plan(),tmp_path/"output.mid")


def test_existing_destination_with_correct_digest_may_be_replaced(fake_engine,tmp_path):
    destination=tmp_path/"package.zip"
    destination.write_bytes(b"old")
    receipt=publish_stem_pack(score(),plan(),destination,
            PublicationOptions(overwrite=True,expected_previous_sha256=sha256(b"old").hexdigest()))
    assert receipt["publication"]["previous_sha256"]==sha256(b"old").hexdigest()
    assert inspect_stem_pack(destination.read_bytes())["stems"]==2


def test_source_digest_unchanged_after_render(fake_engine):
    original=score()
    digest=original.sha256
    render_stem_pack(original,plan())
    assert original.sha256==digest
