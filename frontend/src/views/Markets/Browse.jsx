import { useState, useEffect, useRef, useCallback } from 'react';
import api from '../../api/client';
import AssetModal from '../../components/AssetModal';
import TickerInput from '../../components/TickerInput';

// ─── Formatters ───────────────────────────────────────────────────────────────
const fmtMktCap = (v) => {
  if (v == null) return '—';
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
};
const fmtPct  = (v, raw = false) => v == null ? '—' : raw ? `${v.toFixed(2)}%` : `${(v * 100).toFixed(1)}%`;
const fmtNum  = (v) => v == null ? '—' : v.toFixed(2);
const clr     = (v, goodHigh = true) => {
  if (v == null) return '#64748b';
  return goodHigh ? (v >= 0 ? '#4ade80' : '#f87171') : (v <= 0 ? '#4ade80' : '#f87171');
};

// ─── Pill Filter Button ───────────────────────────────────────────────────────
function Pill({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '0.35rem 0.85rem', borderRadius: '2rem', fontSize: '0.75rem',
      fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s',
      border: active ? '1px solid rgba(99,102,241,0.6)' : '1px solid rgba(255,255,255,0.08)',
      background: active ? 'rgba(99,102,241,0.2)' : 'rgba(255,255,255,0.04)',
      color: active ? '#a5b4fc' : '#64748b',
    }}>
      {label}
    </button>
  );
}

// ─── Sort Header ─────────────────────────────────────────────────────────────
function SortTh({ label, field, sortBy, sortDir, onSort, align = 'right' }) {
  const active = sortBy === field;
  return (
    <th
      onClick={() => onSort(field)}
      style={{
        padding: '0.6rem 0.75rem', fontSize: '0.68rem', fontWeight: 600,
        color: active ? '#a5b4fc' : '#475569', cursor: 'pointer',
        textAlign: align, textTransform: 'uppercase', letterSpacing: '0.05em',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        userSelect: 'none', whiteSpace: 'nowrap',
        transition: 'color 0.15s',
      }}
    >
      {label} {active ? (sortDir === 'desc' ? '↓' : '↑') : ''}
    </th>
  );
}

// ─── Asset Card Row ───────────────────────────────────────────────────────────
function AssetRow({ item, onSelect }) {
  const pct52w = item.fifty_two_week_change;
  return (
    <tr
      onClick={() => onSelect(item.ticker)}
      style={{
        cursor: 'pointer', transition: 'background 0.12s',
        borderBottom: '1px solid rgba(255,255,255,0.03)',
      }}
      onMouseOver={e => e.currentTarget.style.background = 'rgba(99,102,241,0.06)'}
      onMouseOut={e => e.currentTarget.style.background = 'transparent'}
    >
      <td style={{ padding: '0.75rem', paddingLeft: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: 34, height: 34, borderRadius: '0.5rem', flexShrink: 0,
            background: 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2))',
            border: '1px solid rgba(99,102,241,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '0.65rem', fontWeight: 800, color: '#a5b4fc', letterSpacing: '-0.02em',
          }}>
            {item.ticker.slice(0, 4)}
          </div>
          <div>
            <div style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '0.9rem' }}>{item.ticker}</div>
            <div style={{ color: '#475569', fontSize: '0.7rem', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.name || '—'}</div>
          </div>
        </div>
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: '#64748b', fontSize: '0.75rem' }}>
        <span style={{
          padding: '0.15rem 0.5rem', borderRadius: '0.9rem', fontSize: '0.65rem', fontWeight: 700,
          background: item.asset_type === 'ETF' ? 'rgba(245,158,11,0.12)' : 'rgba(99,102,241,0.12)',
          color: item.asset_type === 'ETF' ? '#fbbf24' : '#a5b4fc',
          border: item.asset_type === 'ETF' ? '1px solid rgba(245,158,11,0.2)' : '1px solid rgba(99,102,241,0.2)',
        }}>
          {item.asset_type || 'N/A'}
        </span>
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.82rem' }}>
        {item.sector || '—'}
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: '#e2e8f0', fontWeight: 600, fontSize: '0.82rem' }}>
        {fmtMktCap(item.market_cap)}
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: pct52w != null ? clr(pct52w) : '#64748b', fontWeight: 600, fontSize: '0.82rem' }}>
        {pct52w != null ? `${pct52w >= 0 ? '+' : ''}${(pct52w * 100).toFixed(1)}%` : '—'}
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.82rem' }}>
        {item.trailing_pe != null ? item.trailing_pe.toFixed(1) : '—'}
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', color: item.dividend_yield != null ? '#4ade80' : '#475569', fontSize: '0.82rem' }}>
        {item.dividend_yield != null ? `${item.dividend_yield.toFixed(2)}%` : '—'}
      </td>
      <td style={{ padding: '0.75rem', textAlign: 'right', fontSize: '0.82rem',
        color: item.profit_margins != null ? clr(item.profit_margins) : '#475569' }}>
        {item.profit_margins != null ? fmtPct(item.profit_margins) : '—'}
      </td>
      <td style={{ padding: '0.75rem', paddingRight: '1.25rem', textAlign: 'right', color: '#64748b', fontSize: '0.82rem' }}>
        {fmtNum(item.beta)}
      </td>
    </tr>
  );
}

// ─── Main Markets Page ────────────────────────────────────────────────────────
export default function MarketsPage() {
  const [query, setQuery]           = useState('');
  const [sector, setSector]         = useState('');
  const [assetType, setAssetType]   = useState('');
  const [sortBy, setSortBy]         = useState('market_cap');
  const [sortDir, setSortDir]       = useState('desc');
  const [page, setPage]             = useState(1);

  const [results, setResults]       = useState([]);
  const [total, setTotal]           = useState(0);
  const [sectors, setSectors]       = useState([]);
  const [loading, setLoading]       = useState(true);

  // Modal
  const [selectedTicker, setSelectedTicker] = useState(null);

  const PAGE_SIZE = 24;

  // ── Fetch browse data ──
  const fetchBrowse = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        q: query, sector, asset_type: assetType,
        sort_by: sortBy, sort_dir: sortDir,
        page, page_size: PAGE_SIZE,
      });
      const res = await api.get(`/market/browse?${params}`);
      setResults(res.data.items);
      setTotal(res.data.total);
      if (res.data.sectors?.length) setSectors(res.data.sectors);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [query, sector, assetType, sortBy, sortDir, page]);

  useEffect(() => { fetchBrowse(); }, [fetchBrowse]);

  // Autocomplete & lookup state
  const [lookupStatus, setLookupStatus] = useState(null); // null | 'syncing' | 'invalid'

  const handleSearchValidChange = (ticker, isValid) => {
    if (isValid === true && ticker) {
      // Newly discovered valid ticker — refresh the browse list
      setTimeout(() => fetchBrowse(), 2000);
      setLookupStatus('syncing');
    } else if (isValid === false) {
      setLookupStatus('invalid');
    } else {
      setLookupStatus(null);
    }
  };

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
    setPage(1);
  };

  const handleFilterChange = (fn) => { fn(); setPage(1); };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '1.75rem 2rem', overflow: 'auto', gap: '1.25rem' }}>
      <style>{`
        @keyframes mkFadeIn { from { opacity: 0; transform: translateY(6px) } to { opacity: 1; transform: translateY(0) } }
        .mk-row:hover td { background: rgba(99,102,241,0.04); }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
      `}</style>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ color: '#f1f5f9', fontWeight: 800, fontSize: '1.6rem', margin: 0, letterSpacing: '-0.03em' }}>
            Markets
          </h1>
          <p style={{ color: '#475569', fontSize: '0.8rem', margin: '0.25rem 0 0' }}>
            {total.toLocaleString()} assets tracked · data mining in progress
          </p>
        </div>

        {/* Search + lookup status */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.35rem', width: 340 }}>
          <TickerInput
            value={query}
            onChange={(v) => { setQuery(v.toUpperCase()); setPage(1); setLookupStatus(null); }}
            onValidChange={(ticker, isValid) => {
              handleSearchValidChange(ticker, isValid);
              if (isValid === true && ticker) setSelectedTicker(ticker);
            }}
            placeholder="Search ticker or company..."
            strict={false}
            style={{ width: '100%' }}
            inputStyle={{ textTransform: 'none', fontSize: '0.85rem' }}
          />
          {lookupStatus === 'syncing' && (
            <span style={{ fontSize: '0.7rem', color: '#fbbf24' }}>
              ⚡ New ticker found on Yahoo Finance — syncing to database...
            </span>
          )}
          {lookupStatus === 'invalid' && (
            <span style={{ fontSize: '0.7rem', color: '#f87171' }}>
              ✗ Not a recognised stock or ETF ticker
            </span>
          )}
        </div>
      </div>

      {/* ── Filter Row ── */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
        {/* Asset type pills */}
        <div style={{ display: 'flex', gap: '0.4rem', marginRight: '0.5rem' }}>
          {['', 'EQUITY', 'ETF'].map(t => (
            <Pill key={t} label={t === '' ? 'All Types' : t} active={assetType === t}
              onClick={() => handleFilterChange(() => setAssetType(t))} />
          ))}
        </div>

        <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.08)', margin: '0 0.25rem' }} />

        {/* Sector select */}
        <select
          value={sector}
          onChange={e => handleFilterChange(() => setSector(e.target.value))}
          style={{
            background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '2rem', color: sector ? '#a5b4fc' : '#64748b',
            padding: '0.35rem 1rem', fontSize: '0.75rem', cursor: 'pointer', outline: 'none',
          }}
        >
          <option value="">All Sectors</option>
          {sectors.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        <div style={{ marginLeft: 'auto', color: '#475569', fontSize: '0.75rem' }}>
          Showing {Math.min((page - 1) * PAGE_SIZE + 1, total)}–{Math.min(page * PAGE_SIZE, total)} of {total.toLocaleString()}
        </div>
      </div>

      {/* ── Table ── */}
      <div style={{
        background: 'rgba(10,14,26,0.7)', backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.06)', borderRadius: '1rem',
        overflow: 'hidden', flex: 1,
      }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400, gap: '0.75rem' }}>
            <div style={{ width: 28, height: 28, border: '3px solid rgba(99,102,241,0.15)', borderTopColor: '#6366f1', borderRadius: '50%', animation: 'mkSpin 0.8s linear infinite' }} />
            <style>{`@keyframes mkSpin { to { transform: rotate(360deg) } }`}</style>
            <span style={{ color: '#475569', fontSize: '0.85rem' }}>Loading market data...</span>
          </div>
        ) : results.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 400, gap: '0.5rem' }}>
            <p style={{ fontSize: '2rem' }}>📡</p>
            <p style={{ color: '#475569', fontSize: '0.85rem' }}>No assets match your filters yet. Mining is in progress...</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                  <SortTh label="Asset" field="ticker" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} align="left" />
                  <SortTh label="Type" field="asset_type" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="Sector" field="sector" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="Mkt Cap" field="market_cap" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="52W Chg" field="fifty_two_week_change" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="P/E" field="trailing_pe" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="Div Yield" field="dividend_yield" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="Net Margin" field="profit_margins" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                  <SortTh label="Beta" field="beta" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} />
                </tr>
              </thead>
              <tbody>
                {results.map(item => (
                  <AssetRow key={item.ticker} item={item} onSelect={setSelectedTicker} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Pagination ── */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.4rem' }}>
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{
              padding: '0.4rem 0.9rem', borderRadius: '0.4rem', fontSize: '0.8rem', cursor: page === 1 ? 'default' : 'pointer',
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: page === 1 ? '#334155' : '#94a3b8',
            }}
          >← Prev</button>

          {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
            const p = Math.max(1, Math.min(totalPages - 6, page - 3)) + i;
            return (
              <button key={p} onClick={() => setPage(p)} style={{
                padding: '0.4rem 0.7rem', borderRadius: '0.4rem', fontSize: '0.8rem', cursor: 'pointer',
                background: page === p ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.04)',
                border: page === p ? '1px solid rgba(99,102,241,0.5)' : '1px solid rgba(255,255,255,0.08)',
                color: page === p ? '#a5b4fc' : '#64748b',
              }}>{p}</button>
            );
          })}

          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            style={{
              padding: '0.4rem 0.9rem', borderRadius: '0.4rem', fontSize: '0.8rem', cursor: page === totalPages ? 'default' : 'pointer',
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: page === totalPages ? '#334155' : '#94a3b8',
            }}
          >Next →</button>
        </div>
      )}

      {/* ── Asset Detail Modal ── */}
      {selectedTicker && <AssetModal ticker={selectedTicker} onClose={() => setSelectedTicker(null)} />}
    </div>
  );
}
