"""Tests for the pure integer-EMU geometry helpers."""

import pytest

from edusync_c2.difficulty.geometry import (
    MAX_UNION_SHAPES,
    GroupFrame,
    apply_rotation,
    clamp_box,
    map_through_group,
    to_box,
    to_slide_space,
    union_area,
)


def test_union_overlap_counts_shared_area_once() -> None:
    boxes = [(0, 0, 10, 10), (5, 5, 15, 15), (20, 20, 30, 30)]
    assert union_area(boxes, 100, 100) == 275


def test_union_empty_is_zero() -> None:
    assert union_area([], 100, 100) == 0


def test_union_clamps_to_slide() -> None:
    assert union_area([(-5, -5, 5, 5)], 12, 12) == 25
    assert union_area([(8, 8, 50, 50)], 12, 12) == 16


def test_union_ignores_off_slide_zero_and_inverted_boxes() -> None:
    boxes = [(100, 100, 200, 200), (3, 3, 3, 9), (3, 3, 2, 9), (-20, -20, -10, -10)]
    assert union_area(boxes, 50, 50) == 0


def test_union_merges_overlapping_contained_and_separate_spans() -> None:
    # One column: [0,4] + [2,6] extends, [1,3] is contained, [8,10] is a gap -> 6 + 2 rows.
    boxes = [(0, 0, 10, 4), (0, 2, 10, 6), (0, 1, 10, 3), (0, 8, 10, 10)]
    assert union_area(boxes, 100, 100) == 80


def test_union_cap_falls_back_to_clamped_sum() -> None:
    same = [(0, 0, 10, 10)] * (MAX_UNION_SHAPES + 1)
    assert union_area(same, 100, 100) == 100 * 100  # sum 20100 clamped to the slide area
    thin = [(i, 0, i + 1, 1) for i in range(MAX_UNION_SHAPES + 1)]
    assert union_area(thin, 1000, 1000) == MAX_UNION_SHAPES + 1


def test_union_at_the_cap_is_still_exact() -> None:
    same = [(0, 0, 10, 10)] * MAX_UNION_SHAPES
    assert union_area(same, 100, 100) == 100


def test_clamp_box() -> None:
    assert clamp_box((-1, -1, 5, 5), 10, 10) == (0, 0, 5, 5)
    assert clamp_box((20, 20, 30, 30), 10, 10) is None


def test_to_box() -> None:
    assert to_box((1, 2, 3, 4)) == (1, 2, 4, 6)


def test_map_through_group_scales_and_offsets() -> None:
    frame = GroupFrame(10000, 20000, 4000, 6000, 1000, 1000, 2000, 2000)
    assert map_through_group((1000, 1000, 2000, 2000), frame) == (10000, 20000, 4000, 6000)
    assert map_through_group((2000, 1500, 500, 100), frame) == (12000, 21500, 1000, 300)


def test_map_through_group_floors_integer_division() -> None:
    frame = GroupFrame(0, 0, 10, 10, 0, 0, 3, 3)
    assert map_through_group((1, 1, 1, 1), frame) == (3, 3, 3, 3)


@pytest.mark.parametrize(
    "frame",
    [GroupFrame(0, 0, 1, 1, 0, 0, 0, 5), GroupFrame(0, 0, 1, 1, 0, 0, 5, 0)],
)
def test_map_through_group_zero_child_extent_raises(frame: GroupFrame) -> None:
    with pytest.raises(ValueError, match="zero"):
        map_through_group((0, 0, 1, 1), frame)


def test_nested_groups_compose_innermost_first() -> None:
    inner = GroupFrame(10000, 10000, 4000, 3000, 1000, 1000, 2000, 1000)
    outer = GroupFrame(1000, 2000, 100000, 100000, 0, 0, 100000, 100000)
    rect = apply_rotation((1000, 1000, 2000, 1000), 90)
    assert rect == (1500, 500, 1000, 2000)
    assert to_slide_space(rect, [outer, inner]) == (12000, 10500, 2000, 6000)
    assert to_slide_space(rect, []) == rect


@pytest.mark.parametrize(
    ("degrees", "swapped"),
    [
        (0, False),
        (44.9, False),
        (45, True),
        (90, True),
        (134.9, True),
        (135, False),
        (180, False),
        (225, True),
        (270, True),
        (-90, True),
        (315, False),
        (450, True),
    ],
)
def test_apply_rotation_swaps_for_quarter_turns(degrees: float, swapped: bool) -> None:
    result = apply_rotation((100, 200, 40, 10), degrees)
    assert result == ((115, 185, 10, 40) if swapped else (100, 200, 40, 10))


def test_apply_rotation_keeps_the_centre_to_within_one_emu_with_odd_sizes() -> None:
    x, y, w, h = apply_rotation((0, 0, 5, 2), 90)
    assert (w, h) == (2, 5)
    assert abs((2 * x + w) - 5) <= 1
    assert abs((2 * y + h) - 2) <= 1
