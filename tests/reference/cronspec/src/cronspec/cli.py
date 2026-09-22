import argparse
import sys
from datetime import datetime

from .parser import CronError, parse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cronspec")
    parser.add_argument("expression")
    parser.add_argument("--after")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)
    after = datetime.fromisoformat(args.after) if args.after else datetime.now()
    try:
        times = parse(args.expression).upcoming(after, args.count)
    except CronError as exc:
        print(exc, file=sys.stderr)
        return 2
    for when in times:
        print(when.strftime("%Y-%m-%dT%H:%M"))
    return 0
