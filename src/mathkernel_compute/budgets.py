"""Native-currency admission ledger, scoped to one durable coordinator journal."""
from decimal import Decimal
from .models import Money
from .protocol import canonical, parse


def summary(journal, budget):
    reserved = sum((Decimal(r['amount']) for r in journal.all('budget_reservations')
                    if r.get('budget_id') == budget.budget_id and r['state'] not in {'VOID', 'SETTLED'}), Decimal(0))
    settled = sum((Decimal(r['amount']) for r in journal.all('budget_settlements')
                   if r['budget_id'] == budget.budget_id), Decimal(0))
    return {'budget_id': budget.budget_id, 'account_scope': budget.account_scope, 'currency': 'USD',
            'limit': budget.limit.amount, 'reserved_or_unknown': format(reserved, '.2f'),
            'user_reconciled_spend': format(settled, '.2f'),
            'available': format(Decimal(budget.limit.amount) - reserved - settled, '.2f'),
            'scope': 'this coordinator only; other controllers and external spending are not tracked'}


def reserve(journal, binding):
    current = summary(journal, binding.budget)
    if Decimal(binding.quote.reservation.amount) > Decimal(current['available']):
        raise PermissionError('AGGREGATE_BUDGET_EXCEEDED')


def settle(journal, job_id, amount, source, *, retained_storage_accounted):
    return settle_resource(journal, job_id, journal.get('jobs', job_id)['resources'], amount, source,
                           retained_storage_accounted=retained_storage_accounted)


def settle_resource(journal, job_id, resources, amount, source, *, retained_storage_accounted):
    amount = parse(Money, canonical(amount))
    if amount.currency != 'USD' or Decimal(amount.amount) > Decimal('1000000000000') or not retained_storage_accounted or not isinstance(source, str) or not 1 <= len(source) <= 512:
        raise ValueError('Explicit native USD cost and retained storage accounting required')
    if resources not in {'RELEASE_CONFIRMED', 'NOT_OWNED'}:
        raise PermissionError('RESOURCE_EXPOSURE_UNRESOLVED')
    reservation = journal.get('budget_reservations', job_id)
    if not reservation.get('budget_id'):
        raise ValueError('MANAGED_BUDGET_REQUIRED')
    record = dict(job_id=job_id, budget_id=reservation['budget_id'], account_scope=reservation['account_scope'],
                  amount=amount.amount, currency='USD', source=source,
                  retained_storage_accounted=True, authority='local-host-user-reconciled')
    try:
        old = journal.get('budget_settlements', job_id)
        if old != record:
            raise ValueError('IMMUTABLE_SETTLEMENT_CONFLICT')
    except KeyError:
        journal.put('budget_settlements', job_id, record)
    reservation['state'] = 'SETTLED'
    journal.put('budget_reservations', job_id, reservation)
    return record
