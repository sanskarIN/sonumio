import math

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.room_response import (
    Point3D, ShoeboxRoom, RoomConfig, enumerate_paths, build_room_response,
    render_room,
)


def test_direct_path_delay_gain_and_fractional_centroid():
    room = ShoeboxRoom(5, 6, 3)
    source = Point3D(1, 2, 1)
    receiver = Point3D(2, 2, 1)
    config = RoomConfig(sample_rate=8000, reflection_order=0)
    result = build_room_response(room, (source,), (receiver,), config)
    path, = result.paths
    assert path.direct
    assert path.distance_metres == 1
    assert path.amplitude == 1
    assert path.arrival_frames == 8000 / 343
    h = result.impulse.coefficients[0, 0]
    assert h.sum() == pytest.approx(1)
    assert np.dot(h, np.arange(len(h))) == pytest.approx(path.arrival_frames)
    assert np.flatnonzero(h).tolist() == [23, 24]


def test_first_order_six_independent_walls():
    room = ShoeboxRoom(6, 7, 8, (.1, .2, .3, .4, .5, .6))
    source = Point3D(1, 2, 3)
    receiver = Point3D(2, 3, 4)
    paths = enumerate_paths(room, (source,), (receiver,),
                            RoomConfig(reflection_order=1))
    assert len(paths) == 7
    expected = {
        (-1, 0, 0): ((-1, 2, 3), .1, 0),
        (1, 0, 0): ((11, 2, 3), .2, 1),
        (0, -1, 0): ((1, -2, 3), .3, 2),
        (0, 1, 0): ((1, 12, 3), .4, 3),
        (0, 0, -1): ((1, 2, -3), .5, 4),
        (0, 0, 1): ((1, 2, 13), .6, 5),
    }
    for path in paths[1:]:
        image, coefficient, wall = expected[path.image_index]
        assert path.wall_hits == tuple(int(i == wall) for i in range(6))
        assert path.distance_metres == pytest.approx(math.dist(image, receiver.xyz))
        assert path.amplitude == pytest.approx(coefficient / path.distance_metres)


@pytest.mark.parametrize("order,expected", [(0, 1), (1, 7), (2, 25), (3, 63), (4, 129)])
def test_image_lattice_counts_and_unique_paths(order, expected):
    room = ShoeboxRoom(3, 4, 5)
    p = Point3D(1, 2, 2)
    paths = enumerate_paths(room, (p,), (p,), RoomConfig(reflection_order=order))
    assert len(paths) == expected
    assert len({path.image_index for path in paths}) == expected
    for path in paths:
        assert path.order == sum(map(abs, path.image_index))
        assert path.order <= order


def test_second_order_wall_products_and_distances():
    room = ShoeboxRoom(6, 7, 8, (.2, .3, .4, .5, .6, .7))
    source = Point3D(1, 2, 3)
    receiver = Point3D(2, 3, 4)
    paths = {p.image_index: p for p in enumerate_paths(
        room, (source,), (receiver,), RoomConfig(reflection_order=2)
    )}
    for index, image in [((-2, 0, 0), (-11, 2, 3)), ((2, 0, 0), (13, 2, 3))]:
        p = paths[index]
        assert p.wall_hits == (1, 1, 0, 0, 0, 0)
        assert p.distance_metres == pytest.approx(math.dist(image, receiver.xyz))
        assert p.amplitude == pytest.approx(.06 / p.distance_metres)
    p = paths[(1, -1, 0)]
    assert p.wall_hits == (0, 1, 1, 0, 0, 0)
    assert p.amplitude == pytest.approx(.12 / p.distance_metres)


def test_absorbing_walls_and_no_direct_return_silent_matrix():
    room = ShoeboxRoom(5, 5, 5, (0.,) * 6)
    p = Point3D(2, 2, 2)
    result = build_room_response(room, (p,), (p,), RoomConfig(include_direct=False))
    assert result.paths == ()
    assert result.impulse.coefficients.shape == (1, 1, 1)
    np.testing.assert_array_equal(result.impulse.coefficients, 0)


def test_arrival_cutoff_excludes_out_of_window_paths():
    room = ShoeboxRoom(90, 90, 90)
    p = Point3D(1, 1, 1)
    q = Point3D(80, 80, 80)
    result = build_room_response(room, (p,), (q,), RoomConfig(max_seconds=.01))
    assert result.paths == ()
    assert result.impulse.geometry.taps == 1


def test_multiple_sources_and_receivers_match_path_sums():
    room = ShoeboxRoom(4, 5, 3)
    sources = (Point3D(1, 1, 1), Point3D(3, 4, 1))
    receivers = (Point3D(2, 2, 1), Point3D(3, 2, 1))
    config = RoomConfig(sample_rate=8000, reflection_order=2)
    result = build_room_response(room, sources, receivers, config)
    matrix = result.impulse.coefficients
    for r in range(2):
        for s in range(2):
            selected = [p for p in result.paths if (p.receiver, p.source) == (r, s)]
            assert matrix[r, s].sum() == pytest.approx(sum(p.amplitude for p in selected))
            assert np.dot(matrix[r, s], np.arange(matrix.shape[2])) == pytest.approx(
                sum(p.amplitude * p.arrival_frames for p in selected)
            )
    audio = make_buffer(np.array([[1.], [0.]], dtype=np.float32), 8000)
    rendered = render_room(audio, room, sources, receivers, config)
    np.testing.assert_allclose(rendered.samples, matrix[:, 0], atol=1e-7)


def test_reciprocity_for_equal_source_and_receiver_clocks():
    room = ShoeboxRoom(6, 7, 8)
    a, b = Point3D(1, 2, 3), Point3D(3, 4, 5)
    config = RoomConfig(sample_rate=8000, reflection_order=3)
    forward = build_room_response(room, (a,), (b,), config).impulse.coefficients
    reverse = build_room_response(room, (b,), (a,), config).impulse.coefficients
    np.testing.assert_allclose(forward, reverse, atol=1e-12)


@pytest.mark.parametrize("bad", [
    Point3D(0, 1, 1), Point3D(5, 1, 1), Point3D(1, 1, 99),
])
def test_boundary_and_exterior_rejected(bad):
    room = ShoeboxRoom(5, 5, 5)
    with pytest.raises(AudioContractError):
        enumerate_paths(room, (bad,), (Point3D(1, 1, 1),))


@pytest.mark.parametrize("kwargs", [
    {"reflection_order": 5}, {"reflection_order": True}, {"sample_rate": 2},
    {"max_seconds": float("nan")}, {"include_direct": 1},
    {"speed_of_sound": 0}, {"minimum_distance": 0}, {"block_frames": 63},
])
def test_config_validation(kwargs):
    with pytest.raises(AudioContractError):
        RoomConfig(**kwargs)


@pytest.mark.parametrize("reflections", [(1.,) * 6, (.1,) * 5, [.1] * 6])
def test_invalid_wall_coefficients(reflections):
    with pytest.raises(AudioContractError):
        ShoeboxRoom(3, 4, 5, reflections)


def test_render_rejects_clock_and_source_count():
    room = ShoeboxRoom(4, 4, 4)
    p = Point3D(1, 1, 1)
    audio = make_buffer(np.ones(3), 8000)
    with pytest.raises(AudioContractError):
        render_room(audio, room, (p,), (p,))
    with pytest.raises(AudioContractError):
        render_room(audio, room, (p, p), (p,), RoomConfig(sample_rate=8000))


def test_tap_limit_before_allocation():
    room = ShoeboxRoom(100, 100, 100)
    source = Point3D(1, 1, 1)
    receiver = Point3D(99, 99, 99)
    config = RoomConfig(sample_rate=192000, reflection_order=4, max_seconds=10)
    with pytest.raises(AudioLimitError):
        build_room_response(room, (source,), (receiver,), config)
