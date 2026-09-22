from datetime import datetime, timedelta


class Cron:
    def __init__(self, source, minutes, hours, days, months, weekdays,
                 days_restricted, weekdays_restricted):
        self.source = source
        self.minutes, self.hours, self.days = minutes, hours, days
        self.months, self.weekdays = months, weekdays
        self._either = days_restricted and weekdays_restricted

    def __str__(self):
        return self.source

    def _day(self, when):
        dom = when.day in self.days
        dow = (when.isoweekday() % 7) in self.weekdays
        return (dom or dow) if self._either else (dom and dow)

    def matches(self, when):
        return (when.minute in self.minutes and when.hour in self.hours
                and when.month in self.months and self._day(when))

    def next(self, after):
        from .parser import CronError

        when = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = after + timedelta(days=5 * 366)
        while when <= limit:
            if when.month not in self.months or not self._day(when):
                when = (when + timedelta(days=1)).replace(hour=0, minute=0)
            elif when.hour not in self.hours:
                when = (when + timedelta(hours=1)).replace(minute=0)
            elif when.minute not in self.minutes:
                when += timedelta(minutes=1)
            else:
                return when
        raise CronError(f"{self.source!r} never fires")

    def upcoming(self, after, count):
        times = []
        for _ in range(count):
            after = self.next(after)
            times.append(after)
        return times
