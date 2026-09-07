import type { Plan } from './workflow';
import { Text } from '../app/components';

export function ComputeReview({ plan }: { plan: Plan }) {
  return (
    <section>
      <h4>Resources and cost</h4>
      <p>
        {plan.cost.amount === null
          ? 'Cost unknown'
          : `${plan.cost.amount} ${plan.cost.currency ?? '(currency unknown)'}`}{' '}
        · <Text>{plan.cost.uncertainty}</Text>
      </p>
      <p>
        Price source: <Text>{plan.cost.source}</Text> · observed {plan.cost.observed_at}
      </p>
      <p className="notice">
        An estimate is not a spending cap. This contract does not advertise a hard cap or authorize
        automatic provider fallback.
      </p>
      <table>
        <caption>Host resource estimates</caption>
        <thead>
          <tr>
            <th>Resource</th>
            <th>Amount</th>
            <th>Unit</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {plan.resources.map((r, i) => (
            <tr key={i}>
              <th scope="row">
                <Text>{r.name}</Text>
              </th>
              <td>{r.amount ?? 'Unknown'}</td>
              <td>
                <Text>{r.unit}</Text>
              </td>
              <td>
                <Text>{r.source}</Text>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {plan.cost.exclusions.length > 0 && (
        <>
          <h4>Excluded from estimate</h4>
          <ul>
            {plan.cost.exclusions.map((e, i) => (
              <li key={i}>
                <Text>{e}</Text>
              </li>
            ))}
          </ul>
        </>
      )}
      <h4>Data export closure</h4>
      <p>
        Host-resolved inputs and destinations, including ancestors reported by the host. Imported
        labels cannot authorize these exports.
      </p>
      {plan.exports.length ? (
        <table>
          <thead>
            <tr>
              <th>Input</th>
              <th>Classification</th>
              <th>Destination</th>
              <th>Bytes</th>
            </tr>
          </thead>
          <tbody>
            {plan.exports.map((e, i) => (
              <tr key={i}>
                <td>
                  <Text>{e.input_ref}</Text>
                </td>
                <td>
                  <Text>{e.classification}</Text>
                </td>
                <td>
                  <Text>{e.destination}</Text>
                </td>
                <td>{e.bytes ?? 'Unknown'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No exports reported by this host plan.</p>
      )}
      <h4>Target alternatives</h4>
      <table>
        <thead>
          <tr>
            <th>Target</th>
            <th>Host decision</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {plan.alternatives.map((a, i) => (
            <tr key={i}>
              <td>
                <Text>{a.target}</Text>
              </td>
              <td>{a.accepted ? 'Accepted' : 'Rejected'}</td>
              <td>
                <Text>{a.reason}</Text>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
