"""Held-out acceptance tests for the cronspec task. Never shown to the factory."""
import subprocess
import sys
from datetime import datetime

import pytest

import cronspec
from cronspec import Cron, CronError, parse


def test_every_minute_expands_every_field():
    cron = parse("* * * * *")
    assert isinstance(cron, Cron)
    assert cron.minutes == frozenset(range(60))
    assert cron.hours == frozenset(range(24))
    assert cron.days == frozenset(range(1, 32))
    assert cron.months == frozenset(range(1, 13))
    assert cron.weekdays == frozenset(range(7))


def test_lists_ranges_and_steps():
    cron = parse("0,15,30 9-17/4 */10 1-3 *")
    assert cron.minutes == {0, 15, 30}
    assert cron.hours == {9, 13, 17}
    assert cron.days == {1, 11, 21, 31}
    assert cron.months == {1, 2, 3}


def test_start_with_step_runs_to_the_top_of_the_range():
    assert parse("50/3 * * * *").minutes == {50, 53, 56, 59}


def test_names_are_case_insensitive_and_usable_in_ranges():
    cron = parse("0 0 * jan-Mar MON-fri")
    assert cron.months == {1, 2, 3}
    assert cron.weekdays == {1, 2, 3, 4, 5}


def test_seven_is_sunday():
    assert parse("0 0 * * 7").weekdays == {0}
    assert parse("0 0 * * 5-7").weekdays == {5, 6, 0}


def test_macros():
    assert parse("@hourly").minutes == {0} and parse("@hourly").hours == set(range(24))
    yearly = parse("@annually")
    assert (yearly.minutes, yearly.hours, yearly.days, yearly.months) == (
        {0}, {0}, {1}, {1})
    assert parse("@weekly").weekdays == {0}
    assert parse("@midnight").hours == {0}


def test_str_returns_the_expression():
    assert str(parse("  */5 * * * *  ")) == "*/5 * * * *"
    assert str(parse("@daily")) == "@daily"


def test_extra_whitespace_between_fields():
    assert parse("0  12\t* * *").hours == {12}


def test_cron_error_is_a_value_error():
    assert issubclass(CronError, ValueError)
    assert cronspec.CronError is CronError


@pytest.mark.parametrize("expression, field", [
    ("60 * * * *", "minute"),
    ("* 24 * * *", "hour"),
    ("* * 0 * *", "day of month"),
    ("* * * 13 *", "month"),
    ("* * * * 8", "day of week"),
    ("*/0 * * * *", "minute"),
    ("* 5-2 * * *", "hour"),
    ("* * * FOO *", "month"),
    ("* * 1,,2 * *", "day of month"),
])
def test_malformed_fields_are_named(expression, field):
    with pytest.raises(CronError) as caught:
        parse(expression)
    assert field in str(caught.value)


@pytest.mark.parametrize("expression", ["* * * *", "* * * * * *", "", "@often"])
def test_malformed_expressions(expression):
    with pytest.raises(CronError):
        parse(expression)


def test_matches_ignores_seconds():
    cron = parse("30 14 * * *")
    assert cron.matches(datetime(2026, 5, 4, 14, 30, 59, 999))
    assert not cron.matches(datetime(2026, 5, 4, 14, 31))


def test_next_is_strictly_after():
    cron = parse("30 14 * * *")
    assert cron.next(datetime(2026, 5, 4, 14, 30)) == datetime(2026, 5, 5, 14, 30)
    assert cron.next(datetime(2026, 5, 4, 14, 29, 59)) == datetime(2026, 5, 4, 14, 30)


def test_next_rolls_over_the_year():
    assert parse("@yearly").next(datetime(2026, 12, 31, 23, 59)) == datetime(2027, 1, 1)


def test_next_skips_months_without_the_day():
    assert parse("0 0 31 * *").next(datetime(2026, 4, 1)) == datetime(2026, 5, 31)


def test_leap_day():
    assert parse("0 0 29 2 *").next(datetime(2026, 1, 1)) == datetime(2028, 2, 29)


def test_restricted_day_fields_combine_with_or():
    cron = parse("0 0 13 * FRI")
    # 2026-02-06 is a Friday; 2026-02-13 is a Friday and the 13th.
    assert cron.upcoming(datetime(2026, 2, 1), 3) == [
        datetime(2026, 2, 6), datetime(2026, 2, 13), datetime(2026, 2, 20)]
    assert cron.matches(datetime(2026, 3, 13))  # a Friday the 13th
    assert cron.matches(datetime(2026, 4, 13))  # a Monday


def test_unrestricted_day_of_week_does_not_widen_the_day_of_month():
    assert parse("0 0 13 * *").next(datetime(2026, 2, 1)) == datetime(2026, 2, 13)


def test_stepped_star_counts_as_restricted():
    cron = parse("0 0 */15 * MON")
    # the 1st, the 16th, the 31st, or any Monday: 2026-06-01 is a Monday.
    assert cron.upcoming(datetime(2026, 6, 1), 3) == [
        datetime(2026, 6, 8), datetime(2026, 6, 15), datetime(2026, 6, 16)]


def test_upcoming_in_order():
    times = parse("*/20 9 * * MON-FRI").upcoming(datetime(2026, 9, 18, 9, 30), 4)
    # 2026-09-18 is a Friday.
    assert times == [datetime(2026, 9, 18, 9, 40), datetime(2026, 9, 21, 9, 0),
                     datetime(2026, 9, 21, 9, 20), datetime(2026, 9, 21, 9, 40)]


def test_impossible_schedule_raises():
    with pytest.raises(CronError):
        parse("0 0 30 2 *").next(datetime(2026, 1, 1))


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "cronspec", *args],
                          capture_output=True, text=True, timeout=60)


def test_cli_prints_upcoming_times():
    done = _cli("0 12 * * *", "--after", "2026-03-01T13:00", "--count", "2")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["2026-03-02T12:00", "2026-03-03T12:00"]


def test_cli_defaults_to_five():
    done = _cli("@hourly", "--after", "2026-03-01T13:00")
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.split()) == 5


def test_cli_rejects_a_malformed_expression():
    done = _cli("* * * 13 *", "--after", "2026-03-01T13:00")
    assert done.returncode == 2
    assert "month" in done.stderr
