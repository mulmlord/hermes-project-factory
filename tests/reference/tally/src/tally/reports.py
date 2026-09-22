from .money import Money


def trial_balance(ledger, currency="USD", as_of=None):
    return [(name, ledger.balance(name, currency, as_of, children=False))
            for name in ledger.accounts()]


def income_statement(ledger, start, end, currency="USD"):
    totals = {"income": Money(0, currency), "expense": Money(0, currency)}
    for txn in ledger.transactions():
        if not start <= txn.date <= end:
            continue
        for name, money in txn.postings:
            kind = ledger.kinds[name]
            if kind in totals and money.currency == currency:
                totals[kind] = totals[kind] + money
    income = -totals["income"]
    return {"income": income, "expenses": totals["expense"],
            "net": income - totals["expense"]}
