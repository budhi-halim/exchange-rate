export const EXCHANGE_BASE = 'https://budhi-halim.github.io/exchange-rate/data/';

/** All applications use the published rates, including local previews. */
export async function fetchExchange(file, signal) {
  if (!['today.json', 'history.json'].includes(file)) throw new Error('rate-source');
  const response = await fetch(new URL(file, EXCHANGE_BASE), { signal, cache: 'no-store' });
  if (!response.ok) throw new Error('rate-unavailable');
  return response.json();
}
