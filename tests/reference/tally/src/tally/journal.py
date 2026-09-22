from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from . import accounts
from .money import LedgerError, Money


@dataclass(frozen=True)
class Transaction:
    id: int
    date: date
    description: str
    postings: tuple


class Ledger:
    def __init__(self):
        self.kinds = {}
        self._transactions = []

    def open(self, name, kind):
        accounts.check(name, kind)
        if name in self.kinds:
            raise LedgerError(f"{name} is already open")
        self.kinds[name] = kind

    def accounts(self):
        return sorted(self.kinds)

    def check(self, postings):
        postings = tuple((a, m) for a, m in postings)
        if len(postings) < 2:
            raise LedgerError("a transaction needs two postings")
        sums = defaultdict(lambda: 0)
        for account, money in postings:
            if account not in self.kinds:
                raise LedgerError(f"{account} is not open")
            sums[money.currency] += money.amount
        if any(total != 0 for total in sums.values()):
            raise LedgerError("transaction does not balance")
        return postings

    def post(self, when, description, postings):
        postings = self.check(postings)
        txn = Transaction(len(self._transactions) + 1, when, description, postings)
        self._transactions.append(txn)
        return txn

    def transactions(self, account=None):
        if account is None:
            return list(self._transactions)
        return [t for t in self._transactions
                if any(accounts.within(a, account) for a, _ in t.postings)]

    def balance(self, account, currency="USD", as_of=None, children=True):
        if not any(accounts.within(name, account) for name in self.kinds):
            raise LedgerError(f"no account {account}")
        total = Money(0, currency)
        for txn in self._transactions:
            if as_of is not None and txn.date > as_of:
                continue
            for name, money in txn.postings:
                hit = accounts.within(name, account) if children else name == account
                if hit and money.currency == currency:
                    total = total + money
        return total
