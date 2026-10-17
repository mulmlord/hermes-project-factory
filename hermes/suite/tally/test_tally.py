"""Held-out acceptance tests for the tally task. Never shown to the factory."""
import csv
from datetime import date
from decimal import Decimal

import testkit

import tally
from tally import Ledger, LedgerError, Money, Transaction
from tally.csvio import export_csv, import_csv
from tally.reports import income_statement, trial_balance


def usd(amount):
    return Money(amount)


def books():
    ledger = Ledger()
    for name, kind in (("Assets:Bank", "asset"), ("Assets:Cash", "asset"),
                       ("Liabilities:Card", "liability"), ("Equity:Opening", "equity"),
                       ("Income:Salary", "income"), ("Expenses:Food", "expense"),
                       ("Expenses:Rent", "expense")):
        ledger.open(name, kind)
    ledger.post(date(2026, 1, 1), "opening", [("Assets:Bank", usd("1000")),
                                              ("Equity:Opening", usd("-1000"))])
    ledger.post(date(2026, 1, 5), "pay", [("Assets:Bank", usd("2500")),
                                          ("Income:Salary", usd("-2500"))])
    ledger.post(date(2026, 1, 10), "rent", [("Expenses:Rent", usd("1200")),
                                            ("Assets:Bank", usd("-1200"))])
    ledger.post(date(2026, 2, 2), "lunch", [("Expenses:Food", usd("12.50")),
                                            ("Liabilities:Card", usd("-12.50"))])
    return ledger


def test_money_rounds_half_even_and_prints():
    assert Money("1.005").amount == Decimal("1.00")
    assert Money("1.015").amount == Decimal("1.02")
    assert Money(3).amount == Decimal("3.00")
    assert str(Money("1.5")) == "1.50 USD"
    assert str(-Money("1.5", "EUR")) == "-1.50 EUR"


def test_money_rejects_floats_and_garbage():
    with testkit.raises(TypeError):
        Money(1.5)
    with testkit.raises(LedgerError):
        Money("twelve")


def test_money_arithmetic_and_ordering():
    assert Money("1.10") + Money("2.20") == Money("3.30")
    assert Money("1") - Money("2.5") == Money("-1.5")
    assert Money("1") < Money("1.01") <= Money("1.01")
    assert Money("2") > Money("1") and Money("2") >= Money("2")
    assert Money("1", "EUR") != Money("1", "USD")


def test_money_is_hashable_and_immutable():
    assert len({Money("1"), Money("1.00"), Money("1", "EUR")}) == 2
    with testkit.raises(Exception):
        Money("1").amount = Decimal("2")


def test_mixing_currencies_raises():
    for operation in (lambda: Money("1") + Money("1", "EUR"),
                      lambda: Money("1") - Money("1", "EUR"),
                      lambda: Money("1") < Money("1", "EUR")):
        with testkit.raises(LedgerError):
            operation()


def test_opening_accounts():
    ledger = Ledger()
    ledger.open("Assets:Bank", "asset")
    ledger.open("Assets", "asset")
    assert ledger.accounts() == ["Assets", "Assets:Bank"]
    for name, kind in (("Assets:Bank", "asset"), ("X", "wealth"),
                       ("Assets::Bank", "asset"), (":Bank", "asset")):
        with testkit.raises(LedgerError):
            ledger.open(name, kind)


def test_post_assigns_ids_and_keeps_postings():
    ledger = books()
    first = ledger.transactions()[0]
    assert isinstance(first, Transaction)
    assert [t.id for t in ledger.transactions()] == [1, 2, 3, 4]
    assert first.date == date(2026, 1, 1) and first.description == "opening"
    assert first.postings == (("Assets:Bank", usd("1000")),
                              ("Equity:Opening", usd("-1000")))


@testkit.mark.parametrize("postings", [
    [("Assets:Bank", Money("10"))],
    [("Assets:Bank", Money("10")), ("Income:Salary", Money("-9.99"))],
    [("Assets:Bank", Money("10")), ("Nowhere", Money("-10"))],
    [("Assets:Bank", Money("10")), ("Income:Salary", Money("-10", "EUR"))],
])
def test_invalid_transactions_leave_the_ledger_unchanged(postings):
    ledger = books()
    with testkit.raises(LedgerError):
        ledger.post(date(2026, 3, 1), "bad", postings)
    assert len(ledger.transactions()) == 4
    assert ledger.balance("Assets:Bank") == usd("2300")


def test_multi_currency_transaction_balances_per_currency():
    ledger = books()
    ledger.open("Assets:Euro", "asset")
    ledger.post(date(2026, 3, 1), "fx", [
        ("Assets:Euro", Money("100", "EUR")), ("Equity:Opening", Money("-100", "EUR")),
        ("Assets:Bank", usd("-110")), ("Equity:Opening", usd("110"))])
    assert ledger.balance("Assets", "EUR") == Money("100", "EUR")
    assert ledger.balance("Assets") == usd("2190")


def test_balance_includes_children_unless_asked_not_to():
    ledger = books()
    assert ledger.balance("Assets") == usd("2300")
    assert ledger.balance("Assets:Bank") == usd("2300")
    assert ledger.balance("Expenses") == usd("1212.50")
    ledger.open("Assets", "asset")
    assert ledger.balance("Assets", children=False) == usd("0")


def test_balance_prefix_is_by_segment_not_by_text():
    ledger = books()
    ledger.open("Assets:Bankruptcy", "asset")
    ledger.post(date(2026, 3, 1), "x", [("Assets:Bankruptcy", usd("5")),
                                        ("Equity:Opening", usd("-5"))])
    assert ledger.balance("Assets:Bank") == usd("2300")


def test_balance_as_of_and_unknown_accounts():
    ledger = books()
    assert ledger.balance("Assets:Bank", as_of=date(2026, 1, 5)) == usd("3500")
    assert ledger.balance("Assets:Cash") == usd("0")
    with testkit.raises(LedgerError):
        ledger.balance("Nowhere")


def test_posting_out_of_date_order():
    ledger = books()
    ledger.post(date(2025, 12, 31), "early", [("Assets:Cash", usd("1")),
                                             ("Equity:Opening", usd("-1"))])
    assert ledger.transactions()[-1].id == 5
    assert ledger.balance("Assets:Cash", as_of=date(2026, 1, 1)) == usd("1")


def test_transactions_filtered_by_account_tree():
    ledger = books()
    assert [t.description for t in ledger.transactions("Expenses")] == ["rent", "lunch"]


def test_trial_balance():
    rows = trial_balance(books())
    assert [name for name, _ in rows] == sorted(name for name, _ in rows)
    assert dict(rows)["Assets:Bank"] == usd("2300")
    assert dict(rows)["Assets:Cash"] == usd("0")
    total = usd("0")
    for _, amount in rows:
        total = total + amount
    assert total == usd("0")


def test_trial_balance_as_of():
    rows = dict(trial_balance(books(), as_of=date(2026, 1, 31)))
    assert rows["Expenses:Food"] == usd("0")


def test_income_statement():
    report = income_statement(books(), date(2026, 1, 1), date(2026, 1, 31))
    assert report == {"income": usd("2500"), "expenses": usd("1200"),
                      "net": usd("1300")}
    february = income_statement(books(), date(2026, 2, 1), date(2026, 2, 28))
    assert february["net"] == usd("-12.50")


def test_export_csv(tmp_path):
    path = tmp_path / "out.csv"
    export_csv(books(), path)
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["id", "date", "description", "account", "amount", "currency"]
    assert rows[1] == ["1", "2026-01-01", "opening", "Assets:Bank", "1000.00", "USD"]
    assert len(rows) == 9


def test_csv_round_trip(tmp_path):
    source = books()
    path = tmp_path / "out.csv"
    export_csv(source, path)
    target = Ledger()
    for name in source.accounts():
        target.open(name, "asset" if name.startswith("Assets") else "equity")
    assert import_csv(target, path) == 4
    assert [t.postings for t in target.transactions()] == [
        t.postings for t in source.transactions()]


def test_csv_import_is_all_or_nothing(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(
        "id,date,description,account,amount,currency\n"
        "7,2026-01-01,good,Assets:Bank,5.00,USD\n"
        "7,2026-01-01,good,Equity:Opening,-5.00,USD\n"
        "8,2026-01-02,bad,Assets:Bank,5.00,USD\n"
        "8,2026-01-02,bad,Equity:Opening,-4.00,USD\n")
    ledger = books()
    with testkit.raises(LedgerError):
        import_csv(ledger, path)
    assert len(ledger.transactions()) == 4


def test_exports():
    assert tally.Money is Money and tally.Ledger is Ledger
    assert issubclass(LedgerError, Exception)
