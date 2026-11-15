import numpy as np

from station_check import border_brightness


def test_border_brightness_reads_each_side():
    gray = np.full((100, 100), 50, np.uint8)
    gray[:40, :] = 90
    mean, sides = border_brightness(gray, [0, 0, 100, 100], margin=40)
    assert sides["sus"] == 90 and sides["jos"] == 50
    assert 50 < mean < 90


def test_border_brightness_looks_only_inside_the_zone():
    gray = np.full((100, 100), 200, np.uint8)
    gray[10:60, 20:80] = 70
    mean, sides = border_brightness(gray, [20, 10, 60, 50], margin=10)
    assert mean == 70 and set(sides.values()) == {70.0}
