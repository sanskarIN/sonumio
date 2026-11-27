"""Geometric propagation and deterministic scene rendering invariants."""
import json
from dataclasses import replace
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError,AudioLimitError
from sonumio.audio_core.spatial_motion import (
    Position,MotionKey,Trajectory,Room,Receiver,MovingSource,SpatialScene,
    image_positions,render_spatial_scene,
)


def mono(values,rate=8000):
    return make_buffer(np.asarray(values,dtype=np.float32),rate)


def test_exact_integer_propagation_delay():
    x=np.zeros(80,dtype=np.float32)
    x[10]=.5
    rate=8000
    receiver=Receiver(spacing_m=0,sound_speed=320,directional_mix=0,rolloff=0)
    source=MovingSource(mono(x),Trajectory.stationary(0,4),start_frame=20)
    scene=SpatialScene(rate,300,(source,),receiver,peak_ceiling=None)
    r=render_spatial_scene(scene)
    assert np.argmax(r.audio.samples[0])==130
    assert r.audio.samples[0,130]==pytest.approx(.5/np.sqrt(2))
    np.testing.assert_array_equal(r.audio.samples[0],r.audio.samples[1])
    assert r.report.paths[0].first_arrival_frame==120
    assert r.report.paths[0].last_arrival_frame==199


def test_distance_rolloff_and_exact_peak_policy():
    x=np.ones(100,dtype=np.float32)*.4
    receiver=Receiver(spacing_m=0,sound_speed=320,reference_distance_m=1,rolloff=1)
    near=MovingSource(mono(x),Trajectory.stationary(0,1))
    far=MovingSource(mono(x),Trajectory.stationary(0,4))
    a=render_spatial_scene(SpatialScene(8000,300,(near,),receiver))
    b=render_spatial_scene(SpatialScene(8000,300,(far,),receiver))
    assert a.report.output_peak/b.report.output_peak==pytest.approx(4)
    loud=SpatialScene(8000,300,(near,)*8,receiver,peak_ceiling=.2)
    r=render_spatial_scene(loud)
    assert r.report.attenuation<1
    assert r.report.output_peak<=.2000001


def test_mirror_stereo_and_heading_rotation():
    x=mono(np.sin(np.arange(200)*.2)*.2)
    source=MovingSource(x,Trajectory.stationary(2,0))
    mirrored=replace(source,trajectory=Trajectory.stationary(-2,0))
    a=render_spatial_scene(SpatialScene(8000,300,(source,)))
    b=render_spatial_scene(SpatialScene(8000,300,(mirrored,)))
    np.testing.assert_allclose(a.audio.samples,b.audio.samples[::-1],atol=1e-7)
    rotated=render_spatial_scene(SpatialScene(8000,300,(source,),Receiver(heading_degrees=180)))
    np.testing.assert_allclose(a.audio.samples,rotated.audio.samples[::-1],atol=1e-7)
    assert np.max(np.abs(a.audio.samples[1]))>np.max(np.abs(a.audio.samples[0]))


def test_moving_source_arrival_clock_contract():
    x=mono(np.sin(np.arange(8001)*2*np.pi*200/8000)*.2)
    trajectory=Trajectory((MotionKey(0,Position(0,10)),MotionKey(8000,Position(0,2))))
    source=MovingSource(x,trajectory)
    receiver=Receiver(spacing_m=0,sound_speed=320,directional_mix=0,rolloff=0)
    r=render_spatial_scene(SpatialScene(8000,8500,(source,),receiver))
    path=r.report.paths[0]
    assert path.minimum_arrival_step==pytest.approx(.975,abs=1e-10)
    assert path.maximum_arrival_step==pytest.approx(.975,abs=1e-10)
    assert path.first_arrival_frame==250
    assert path.last_arrival_frame==8050
    middle=r.audio.samples[0,1000:7000]
    f=np.argmax(np.abs(np.fft.rfft(middle*np.hanning(len(middle)))))*8000/len(middle)
    assert f==pytest.approx(200/.975,abs=2)


def test_first_order_reflection_geometry():
    room=Room(10,10,left=.5,right=0,bottom=0,top=0)
    receiver=Receiver(Position(2,5),spacing_m=0,sound_speed=320,rolloff=0,directional_mix=0)
    x=np.zeros(60,dtype=np.float32)
    x[0]=.8
    source=MovingSource(mono(x),Trajectory.stationary(4,5))
    r=render_spatial_scene(SpatialScene(8000,250,(source,),receiver,room,peak_ceiling=None))
    direct,reflection=r.audio.samples[0,50],r.audio.samples[0,150]
    assert direct==pytest.approx(.8/np.sqrt(2))
    assert reflection==pytest.approx(direct*.5)
    assert [p.path for p in r.report.paths]==["direct","direct","left","left"]
    assert r.report.paths[2].minimum_distance_m==6


@pytest.mark.parametrize("wall,expected",[
    ("left",(-2,3)),("right",(18,3)),("bottom",(2,-3)),("top",(2,17)),
])
def test_image_source_positions(wall,expected):
    x,y=image_positions(np.array([2.]),np.array([3.]),Room(10,10),wall)
    assert (x[0],y[0])==expected


def test_receiver_tail_truncation_report():
    source=MovingSource(mono(np.ones(100)*.1),Trajectory.stationary(0,4))
    r=render_spatial_scene(SpatialScene(8000,50,(source,),Receiver(spacing_m=0,sound_speed=320)))
    assert all(p.truncated and p.rendered_frames==0 for p in r.report.paths)
    assert not np.any(r.audio.samples)


def test_deterministic_source_digest_and_no_mutation():
    source=MovingSource(mono(np.random.default_rng(5).normal(size=300)*.1),Trajectory.stationary(1,1))
    before=source.audio.samples.tobytes()
    scene=SpatialScene(8000,500,(source,))
    a=render_spatial_scene(scene)
    b=render_spatial_scene(scene)
    np.testing.assert_array_equal(a.audio.samples,b.audio.samples)
    assert a.report==b.report
    assert source.audio.samples.tobytes()==before
    assert a.report.scene_sha256!=render_spatial_scene(replace(scene,peak_ceiling=.5)).report.scene_sha256
    json.dumps(a.report.as_dict(),allow_nan=False)


def test_trajectory_and_room_rejections():
    with pytest.raises(AudioContractError):
        Trajectory((MotionKey(1,Position(0,0)),))
    with pytest.raises(AudioContractError):
        Trajectory((MotionKey(0,Position(0,0)),MotionKey(0,Position(0,1))))
    with pytest.raises(AudioContractError,match="speed"):
        MovingSource(mono(np.zeros(100)),Trajectory((MotionKey(0,Position(0,0)),MotionKey(99,Position(0,100)))))
    with pytest.raises(AudioContractError,match="outside"):
        MovingSource(mono(np.zeros(100)),Trajectory((MotionKey(0,Position(0,0)),MotionKey(100,Position(0,0)))))
    source=MovingSource(mono(np.zeros(100)),Trajectory.stationary(-1,1))
    with pytest.raises(AudioContractError,match="outside reflection room"):
        SpatialScene(8000,200,(source,),room=Room(5,5))
    with pytest.raises(AudioContractError,match="leaves"):
        SpatialScene(8000,200,(source,),receiver=Receiver(Position(2,2)),room=Room(5,5))


def test_source_clock_and_layout_rejections():
    with pytest.raises(AudioContractError):
        MovingSource(mono(np.zeros((2,100))),Trajectory.stationary(1,1))
    source=MovingSource(mono(np.zeros(100)),Trajectory.stationary(1,1))
    with pytest.raises(AudioContractError,match="clock"):
        SpatialScene(24000,200,(source,))


def test_budgets_precede_render(monkeypatch):
    import sonumio.audio_core.spatial_motion as module
    source=MovingSource(mono(np.zeros(100)),Trajectory.stationary(1,1))
    scene=SpatialScene(8000,200,(source,))
    monkeypatch.setattr(module,"MAX_PATH_FRAMES",1)
    with pytest.raises(AudioLimitError,match="path-frame"):
        render_spatial_scene(scene)
    monkeypatch.setattr(module,"MAX_PATH_FRAMES",100000)
    monkeypatch.setattr(module,"MAX_WORK_BYTES",1)
    with pytest.raises(AudioLimitError,match="working-memory"):
        render_spatial_scene(scene)
