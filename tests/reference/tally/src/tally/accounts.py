from .money import LedgerError

KINDS = frozenset({"asset", "liability", "equity", "income", "expense"})


def check(name, kind):
    if kind not in KINDS:
        raise LedgerError(f"unknown kind {kind!r}")
    if not name or any(not part for part in name.split(":")):
        raise LedgerError(f"malformed account name {name!r}")


def within(name, root):
    return name == root or name.startswith(root + ":")
