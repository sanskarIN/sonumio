"""Moving room snapshot crossfades and bounded geometry invariants."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.room_response import (
    Point3D, ShoeboxRoom, RoomConfig, render_room,
    build_room_response,
)
from sonumio.audio_core.moving_room import (
    SceneKeyframe, MotionRoomPlan, MotionRoomResult, snapshot_weights,
    render_motion_room,
)


ROOM = ShoeboxRoom(4., 5., 3.)
CONFIG = RoomConfig(sample_rate=8000, reflection_order=0, max_seconds=.25,
                    block_frames=64)
SOURCE_A = Point3D(1., 2., 1.)
SOURCE_B = Point3D(3., 2., 1.)
RECEIVER = Point3D(2., 2., 1.)


def motion(keys=(0,), *, sources=(SOURCE_A,), receivers=(RECEIVER,),
           config=CONFIG, engine="partitioned"):
    entries = tuple(SceneKeyframe(n, sources, receivers) for n in keys)
    return MotionRoomPlan(ROOM, config, entries, engine)


@pytest.mark.parametrize("positions", [
    (0,), (0, 1), (0, 17), (0, 7, 13, 79),
    (0, 64, 128, 255),
])
def test_weights_are_nonnegative_partition_of_unity(positions):
    frames = 300
    values = [snapshot_weights(positions, i, frames)
              for i in range(len(positions))]
    summed = np.sum(values, axis=0)
    np.testing.assert_allclose(summed, 1, atol=1e-14)
    for weights in values:
        assert weights.shape == (frames,)
        assert np.all(np.isfinite(weights))
        assert np.min(weights) >= 0
        assert np.max(weights) <= 1
    assert values[0][0] == 1
    assert values[-1][-1] == 1


@pytest.mark.parametrize("engine", ["matrix", "partitioned"])
@pytest.mark.parametrize("include_tail", [False, True])
def test_one_snapshot_equals_static_reference(engine, include_tail):
    rng = np.random.default_rng(12)
    audio = make_buffer(rng.normal(0, .1, 323).astype(np.float32), 8000)
    plan = motion(engine=engine)
    result = render_motion_room(audio, plan, include_tail=include_tail)
    reference = render_room(audio, ROOM, (SOURCE_A,), (RECEIVER,), CONFIG,
                            include_tail=include_tail, engine=engine)
    assert result.audio.spec == reference.spec
    np.testing.assert_allclose(result.audio.samples, reference.samples, atol=1e-7)
    assert result.report.max_impulse_taps >= 1
    assert result.report.output_frames == reference.frames
    assert len(result.report.snapshots) == 1
    assert result.report.snapshots[0].path_count == 1
    assert result.report.snapshots[0].direct_paths == 1


@pytest.mark.parametrize("engine", ["matrix", "partitioned"])
def test_repeated_identical_geometry_is_static_invariant(engine):
    rng = np.random.default_rng(36)
    audio = make_buffer(rng.normal(0, .1, 435).astype(np.float32), 8000)
    plan = motion((0, 17, 180, 430), engine=engine)
    result = render_motion_room(audio, plan)
    original = render_room(audio, ROOM, (SOURCE_A,), (RECEIVER,), CONFIG,
                           engine=engine)
    assert result.audio.spec == original.spec
    np.testing.assert_allclose(result.audio.samples, original.samples, atol=1e-7)


@pytest.mark.parametrize("engine", ["matrix", "partitioned"])
def test_moving_source_changes_expected_room_amplitude(engine):
    data = np.zeros(420, dtype=np.float32)
    data[100] = 1.
    data[300] = 1.
    audio = make_buffer(data, 8000)
    key_a = SceneKeyframe(0, (Point3D(1., 2., 1.),), (RECEIVER,))
    key_b = SceneKeyframe(210, (Point3D(2.5, 2., 1.),), (RECEIVER,))
    result = render_motion_room(
        audio, MotionRoomPlan(ROOM, CONFIG, (key_a, key_b), engine),
    )
    assert result.audio.frames >= 420
    assert float(np.max(np.abs(result.audio.samples))) > 0
    assert result.report.snapshots[0].impulse_taps != (
        result.report.snapshots[1].impulse_taps
    )
    assert result.report.input_channels == 1
    assert result.report.output_channels == 1
    assert result.report.as_dict()["doppler_simulated"] is False


def test_two_sources_and_receivers_keep_explicit_layout():
    config = RoomConfig(sample_rate=8000, reflection_order=0, max_seconds=.4,
                        block_frames=64)
    sources = (SOURCE_A, SOURCE_B)
    receivers = (Point3D(1.5, 2., 1.), Point3D(2.5, 2., 1.))
    rng = np.random.default_rng(511)
    audio = make_buffer(rng.normal(0, .01, (2, 260)).astype(np.float32), 8000)
    plan = motion((0, 90), sources=sources, receivers=receivers,
                  config=config)
    out = render_motion_room(audio, plan, include_tail=False)
    assert out.audio.channels == 2
    assert out.audio.frames == audio.frames
    assert out.audio.spec.layout == AudioLayout.for_channels(2)
    assert np.isfinite(out.audio.samples).all()
    assert all(stat.input_channels == stat.output_channels == 2
               for stat in out.report.snapshots)


def test_direct_paths_disabled_and_no_reflections_is_silent():
    room = ShoeboxRoom(4, 5, 3, reflections=(0.,) * 6)
    config = RoomConfig(sample_rate=8000, reflection_order=0,
                        include_direct=False, block_frames=64)
    key = SceneKeyframe(0, (SOURCE_A,), (RECEIVER,))
    plan = MotionRoomPlan(room, config, (key,))
    audio = make_buffer(np.ones(200, dtype=np.float32) * .1, 8000)
    result = render_motion_room(audio, plan)
    np.testing.assert_array_equal(result.audio.samples, 0)
    assert result.report.snapshots[0].path_count == 0
    assert result.report.max_impulse_taps == 1


def test_moving_room_preserves_source_and_immutable_output():
    rng = np.random.default_rng(67)
    audio = make_buffer(rng.normal(0, .1, 233).astype(np.float32), 8000)
    original = audio.samples.copy()
    plan = motion((0, 50, 200))
    out = render_motion_room(audio, plan)
    np.testing.assert_array_equal(audio.samples, original)
    with pytest.raises(ValueError):
        out.audio.samples[0, 0] = 1
    assert out.report.as_dict()["model_inference"] is False
    assert out.report.as_dict()["sample_clock_changed"] is False


@pytest.mark.parametrize("frames", [
    (10,), (0, 0), (0, 2, 1), (0,) * 17,
])
def test_invalid_keyframe_sequences_rejected(frames):
    with pytest.raises(AudioContractError):
        motion(frames)


def test_invalid_point_and_topology_rejected():
    with pytest.raises(AudioContractError):
        SceneKeyframe(-1, (SOURCE_A,), (RECEIVER,))
    with pytest.raises(AudioContractError):
        SceneKeyframe(0, (), (RECEIVER,))
    with pytest.raises(AudioContractError):
        MotionRoomPlan(
            ROOM, CONFIG, (
                SceneKeyframe(0, (SOURCE_A,), (RECEIVER,)),
                SceneKeyframe(10, (SOURCE_A, SOURCE_B), (RECEIVER,)),
            )
        )
    with pytest.raises(AudioContractError):
        MotionRoomPlan(
            ROOM, CONFIG, (SceneKeyframe(
                0, (Point3D(4., 2., 1.),), (RECEIVER,)
            ),)
        )
    with pytest.raises(AudioContractError):
        motion(engine="automatic")


def test_source_sample_clock_and_key_limit_rejected():
    audio = make_buffer(np.ones(100, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError):
        render_motion_room(audio, motion((0, 100)))
    with pytest.raises(AudioContractError):
        render_motion_room(make_buffer(np.ones(100, dtype=np.float32), 16000),
                           motion())
    with pytest.raises(AudioContractError):
        render_motion_room(audio, motion(), include_tail=1)
    with pytest.raises(AudioContractError):
        render_motion_room(make_buffer(np.ones((2, 100)), 8000), motion())


@pytest.mark.parametrize("positions,index,frames", [
    ((1,), 0, 100),
    ((0, 0), 0, 100),
    ((0, 50, 25), 1, 100),
    ((0, 50), 2, 100),
    ((0, 100), 1, 100),
    ((0,), True, 100),
    ((0,), 0, 0),
])
def test_invalid_weights_rejected(positions, index, frames):
    with pytest.raises(AudioContractError):
        snapshot_weights(positions, index, frames)
