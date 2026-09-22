from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation


class LedgerError(Exception):
    pass


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __init__(self, amount, currency="USD"):
        if isinstance(amount, float):
            raise TypeError("Money does not accept floats")
        try:
            value = Decimal(amount)
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise LedgerError(f"not an amount: {amount!r}") from exc
        if not value.is_finite():
            raise LedgerError(f"not an amount: {amount!r}")
        object.__setattr__(self, "amount",
                           value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
        object.__setattr__(self, "currency", currency)

    def _same(self, other):
        if not isinstance(other, Money) or other.currency != self.currency:
            raise LedgerError("cannot mix currencies")
        return other.amount

    def __add__(self, other):
        return Money(self.amount + self._same(other), self.currency)

    def __sub__(self, other):
        return Money(self.amount - self._same(other), self.currency)

    def __neg__(self):
        return Money(-self.amount, self.currency)

    def __lt__(self, other):
        return self.amount < self._same(other)

    def __le__(self, other):
        return self.amount <= self._same(other)

    def __gt__(self, other):
        return self.amount > self._same(other)

    def __ge__(self, other):
        return self.amount >= self._same(other)

    def __str__(self):
        return f"{self.amount:.2f} {self.currency}"
