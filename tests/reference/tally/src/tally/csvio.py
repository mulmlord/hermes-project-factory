import csv
from datetime import date

from .money import LedgerError, Money

HEADER = ["id", "date", "description", "account", "amount", "currency"]


def export_csv(ledger, path):
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(HEADER)
        for txn in ledger.transactions():
            for account, money in txn.postings:
                out.writerow([txn.id, txn.date.isoformat(), txn.description, account,
                              f"{money.amount:.2f}", money.currency])


def import_csv(ledger, path):
    groups = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["id"]
            if key not in groups:
                groups[key] = (date.fromisoformat(row["date"]), row["description"], [])
            groups[key][2].append((row["account"], Money(row["amount"], row["currency"])))
    checked = [(when, text, ledger.check(postings))
               for when, text, postings in groups.values()]
    for when, text, postings in checked:
        ledger.post(when, text, postings)
    return len(checked)
