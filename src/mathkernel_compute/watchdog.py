"""One-shot watchdog for an independently installed user-owned control host.

Tickets can be armed before launch. Only unique authenticated ownership matches
can establish an ID; no broad deletion, provisioning, math or artifact admission.
An operator/system scheduler must run tick periodically on a different failure
domain. The library cannot attest that physical deployment or guarantee expiry.
"""
import secrets
from .lambda_models import WatchdogTicket
from .lambda_api import LambdaAPI
from .protocol import canonical, parse
from .provisioning import find_instance, now_ms


class LambdaWatchdog:
    def __init__(self, journal, profiles, *, _http=None):
        self.journal, self.profiles, self.http = journal, profiles, _http

    def _api(self, plan):
        profile = self.profiles.get(plan.profile.target_id)
        if profile is None or profile.account_scope != plan.profile.account_scope:
            raise PermissionError('WATCHDOG_ACCOUNT_NOT_CONFIGURED')
        return LambdaAPI(profile, _http=self.http, journal=self.journal)

    def _arm(self, ticket, *, ticket_digest, independent_host_acknowledged):
        ticket = parse(WatchdogTicket, canonical(ticket))
        if ticket_digest != ticket.digest or independent_host_acknowledged is not True:
            raise PermissionError('WATCHDOG_HOST_APPROVAL_REQUIRED')
        self._api(ticket.plan)  # The host chooses credentials; ticket data cannot choose a key.
        identity = ticket.plan.lease_id
        with self.journal.transaction():
            try:
                old = self.journal.get('watchdog_leases', identity)
                if old['ticket'] != ticket.model_dump(mode='json'):
                    raise PermissionError('WATCHDOG_TICKET_IMMUTABLE')
                return old
            except KeyError:
                if len(self.journal.all('watchdog_leases')) >= 1000:
                    raise ValueError('WATCHDOG_CAPACITY')
                record = {'lease_id': identity, 'ticket': ticket.model_dump(mode='json'),
                    'instance_id': None, 'resources': 'CLEANUP_UNKNOWN', 'next_check_ms': 0,
                    'termination_intents': 0, 'authority': 'independent-host-operator',
                    'deployment_independence': 'operator-acknowledged-not-machine-attested',
                    'message': 'armed; missing instance is not proof of non-creation'}
                self.journal.put('watchdog_leases', identity, record)
                return record

    def tick(self):
        results = []
        for row in self.journal.all('watchdog_leases'):
            plan = parse(WatchdogTicket, canonical(row['ticket'])).plan
            if (row['resources'] == 'RELEASE_CONFIRMED' or now_ms() < plan.cleanup_after_ms
                    or (not self.http and now_ms() < row['next_check_ms'])):
                results.append(row)
                continue
            try:
                api = self._api(plan)
                instance = find_instance(api, plan, row['instance_id'])
                row['instance_id'] = instance['id']
                if instance['status'] == 'terminated':
                    row.update(resources='RELEASE_CONFIRMED', message='provider termination observed; billing not settled')
                elif instance['status'] == 'terminating':
                    row.update(resources='RELEASE_REQUESTED', message='provider termination pending')
                else:
                    with self.journal.transaction():
                        row.update(resources='RELEASE_REQUESTED', message='termination intent committed',
                                   termination_intents=row['termination_intents'] + 1,
                                   next_check_ms=now_ms() + 30000 + secrets.randbelow(15000))
                        self.journal.put('watchdog_leases', row['lease_id'], row)
                    api.call('terminate', {'instance_id': row['instance_id']})
                    row['message'] = 'termination acknowledged; confirmation still required'
            except (OSError, ValueError, KeyError, TypeError, PermissionError):
                row.update(resources='CLEANUP_UNKNOWN', message='provider ownership or termination unresolved; no replacement')
            row['next_check_ms'] = now_ms() + 30000 + secrets.randbelow(15000)
            with self.journal.transaction():
                self.journal.put('watchdog_leases', row['lease_id'], row)
            results.append(row)
        return results
