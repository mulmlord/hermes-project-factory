# tally — requirements

Build `tally`, an installable Python library (Python 3.10+, standard library
only) for double-entry bookkeeping.

## Layout

The import package is `tally`, placed at `src/tally/`, with these modules:

- `tally/__init__.py` — re-exports `Money`, `Ledger`, `Transaction` and
  `LedgerError`
- `tally/money.py` — `Money` and the `LedgerError` exception
- `tally/accounts.py` — account kinds and the account hierarchy
- `tally/journal.py` — `Ledger` and `Transaction`
- `tally/reports.py` — `trial_balance` and `income_statement`
- `tally/csvio.py` — `export_csv` and `import_csv`

Ship a `pyproject.toml` so `pip install .` installs the package, and a test
suite of your own under `tests/`.

## Money

`Money(amount, currency="USD")`

- `amount` may be a `str`, an `int` or a `decimal.Decimal`. A `float` raises
  `TypeError`, because a float cannot hold a cent exactly. A string that is
  not a number raises `LedgerError`.
- The amount is stored as a `Decimal` rounded to two places with
  `ROUND_HALF_EVEN`, and exposed as the attribute `amount`; the currency code
  is exposed as `currency`.
- `Money` is immutable and hashable. Two values are equal when both amount
  and currency are equal.
- `+`, `-` (binary and unary), `<`, `<=`, `>`, `>=` work between values of one
  currency. Mixing currencies in any of them raises `LedgerError`.
- `str(Money("1.5"))` is `"1.50 USD"`; negative values print as `"-1.50 USD"`.

`LedgerError` is a subclass of `Exception`.

## Accounts

Account names are hierarchical, with segments joined by colons:
`Assets:Bank:Checking`. Every account has a kind, one of `asset`,
`liability`, `equity`, `income`, `expense`.

`Ledger.open(name, kind)` opens an account. Opening a name twice, an unknown
kind, or a name with an empty segment (`"Assets::Bank"`, `":Bank"`) raises
`LedgerError`. Opening a child does not open its parents.

`Ledger.accounts()` returns the opened names, sorted.

## Transactions

`Ledger.post(date, description, postings) -> Transaction`

- `date` is a `datetime.date`; `postings` is a list of `(account, Money)`
  pairs.
- There must be at least two postings, every account must be open, and the
  amounts must sum to zero **separately for each currency**. Any violation
  raises `LedgerError` and leaves the ledger unchanged.
- Transactions may be posted in any date order.

A `Transaction` has `id` (1 for the first posted, then 2, 3, ...), `date`,
`description` and `postings`, a tuple of `(account, Money)` pairs in the order
given.

`Ledger.transactions(account=None)` returns transactions in posting order;
with an account, only those touching that account or one of its descendants.

## Balances

`Ledger.balance(account, currency="USD", as_of=None, children=True) -> Money`

sums the postings in `currency` to `account` — and, when `children` is true,
to every account whose name starts with `account + ":"` — from transactions
dated on or before `as_of` (all of them when it is `None`). Sign convention:
the raw sum, so debits are positive and credits negative. The account need
not be open itself (so `balance("Assets")` works when only `Assets:Bank` is
open), but a name matching no opened account, as itself or as a parent,
raises `LedgerError`.

## Reports

`reports.trial_balance(ledger, currency="USD", as_of=None)` returns a list of
`(account, Money)` pairs, one per opened account in name order, each the
account's balance **without** children. The amounts sum to zero.

`reports.income_statement(ledger, start, end, currency="USD")` returns a dict
with keys `income`, `expenses` and `net`, covering transactions dated from
`start` to `end` inclusive. `income` is the total of every `income` account
with its sign flipped, so money earned is positive; `expenses` is the total of
every `expense` account as it stands; `net` is `income - expenses`.

## CSV

`csvio.export_csv(ledger, path)` writes one row per posting with the header
`id,date,description,account,amount,currency`, dates as `YYYY-MM-DD`, amounts
with two decimal places, rows in posting order.

`csvio.import_csv(ledger, path) -> int` reads that format into an existing
ledger whose accounts are already open, grouping rows by `id` into
transactions (ids in the file are not reused; the ledger assigns its own) and
returning the number of transactions posted. If any transaction in the file is
invalid, raise `LedgerError` and post none of them.
