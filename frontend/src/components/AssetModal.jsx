import { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts';
import api from '../api/client';

// ─── Formatters ───────────────────────────────────────────────────────────────
const fmtNum = (v) => {
  if (v == null) return '—';
  if (Math.abs(v) >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (Math.abs(v) >= 1e9)  return `$${(v / 1e9).toFixed(2)}B`;
  if (Math.abs(v) >= 1e6)  return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toLocaleString()}`;
};
const fmtPct = (v) => v == null ? '—' : `${(v * 100).toFixed(2)}%`;
const fmtMultiple = (v) => v == null ? '—' : `${v.toFixed(2)}x`;
const fmtRaw = (v) => v == null ? '—' : v.toFixed(2);

// ─── Badge Chip ───────────────────────────────────────────────────────────────
function Badge({ label, value, sentiment }) {
  const colors = sentiment === 'positive'
    ? { bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.25)', text: '#4ade80' }
    : sentiment === 'negative'
    ? { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.25)', text: '#f87171' }
    : { bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.15)', text: '#94a3b8' };
  return (
    <div style={{
      background: colors.bg,
      border: `1px solid ${colors.border}`,
      borderRadius: '0.5rem',
      padding: '0.6rem 0.85rem',
      display: 'flex', flexDirection: 'column', gap: '0.2rem',
    }}>
      <span style={{ color: '#64748b', fontSize: '0.65rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </span>
      <span style={{ color: colors.text, fontSize: '0.95rem', fontWeight: 700 }}>
        {value}
      </span>
    </div>
  );
}

// ─── Tab Button ───────────────────────────────────────────────────────────────
function TabBtn({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '0.4rem 1rem',
      borderRadius: '0.4rem',
      fontSize: '0.8rem',
      fontWeight: 600,
      cursor: 'pointer',
      border: active ? '1px solid rgba(99,102,241,0.5)' : '1px solid transparent',
      background: active ? 'rgba(99,102,241,0.15)' : 'transparent',
      color: active ? '#a5b4fc' : '#64748b',
      transition: 'all 0.15s',
    }}>
      {label}
    </button>
  );
}

// ─── History Chart ────────────────────────────────────────────────────────────
function HistoryTab({ ticker }) {
  const [data, setData] = useState(null);
  const [range, setRange] = useState('1y');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const res = await api.get(`/market/history?ticker=${ticker}&range=${range}`);
        setData(res.data);
      } catch { setData(null); }
      finally { setLoading(false); }
    })();
  }, [ticker, range]);

  if (loading) return (
    <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={spinnerStyle} />
    </div>
  );

  if (!data?.history) return (
    <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '0.85rem' }}>
      No price history available yet — data mining in progress.
    </div>
  );

  const isUp = data.percent_change >= 0;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <span style={{ color: isUp ? '#4ade80' : '#f87171', fontWeight: 700, fontSize: '1rem' }}>
          {isUp ? '+' : ''}{data.percent_change}%
          <span style={{ color: '#64748b', fontWeight: 400, fontSize: '0.75rem', marginLeft: '0.4rem' }}>{range.toUpperCase()} return</span>
        </span>
        <div style={{ display: 'flex', gap: '0.4rem' }}>
          {['1y', '3y', '5y', 'max'].map(r => (
            <button key={r} onClick={() => setRange(r)} style={{
              padding: '0.25rem 0.55rem', borderRadius: '0.3rem', fontSize: '0.7rem', cursor: 'pointer',
              background: range === r ? 'rgba(99,102,241,0.2)' : 'rgba(255,255,255,0.04)',
              border: range === r ? '1px solid rgba(99,102,241,0.4)' : '1px solid transparent',
              color: range === r ? '#a5b4fc' : '#64748b',
            }}>{r.toUpperCase()}</button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data.history} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
          <defs>
            <linearGradient id="gradAsset" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={isUp ? '#22c55e' : '#ef4444'} stopOpacity={0.35} />
              <stop offset="100%" stopColor={isUp ? '#22c55e' : '#ef4444'} stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="date" stroke="#334155" fontSize={9} tickLine={false}
            tickFormatter={(t) => { const d = new Date(t); return `${d.getMonth()+1}/${String(d.getFullYear()).slice(2)}`; }} />
          <YAxis stroke="#334155" fontSize={9} tickLine={false} width={55}
            tickFormatter={(v) => `$${v >= 1000 ? (v/1000).toFixed(0)+'k' : v.toFixed(0)}`}
            domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '0.5rem', fontSize: '0.8rem' }}
            formatter={(v) => [`$${v.toFixed(2)}`, 'Price']}
            labelFormatter={(l) => new Date(l).toLocaleDateString()}
          />
          <Area type="monotone" dataKey="adj_close" stroke={isUp ? '#22c55e' : '#ef4444'} strokeWidth={1.5} fill="url(#gradAsset)" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Financials Tab ───────────────────────────────────────────────────────────
function FinancialsTab({ statements, actions }) {
  const REVENUE_COLOR = '#6366f1';
  const NET_COLOR = '#22c55e';

  if (!statements?.length) return (
    <div style={{ color: '#475569', fontSize: '0.85rem', textAlign: 'center', padding: '3rem 0' }}>
      No financial statements available yet.
    </div>
  );

  const chartData = [...statements].reverse().map(s => ({
    date: s.report_date?.slice(0, 7) ?? s.report_date,
    revenue: s.total_revenue,
    net: s.net_income,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Revenue & Net Income Bar Chart */}
      <div>
        <p style={{ color: '#94a3b8', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Revenue vs Net Income
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={chartData} margin={{ top: 5, right: 5, left: -10, bottom: 0 }} barCategoryGap="35%">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis dataKey="date" stroke="#334155" fontSize={9} tickLine={false} />
            <YAxis stroke="#334155" fontSize={9} tickLine={false} width={60} tickFormatter={(v) => fmtNum(v).replace('$', '')} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '0.5rem', fontSize: '0.8rem' }}
              formatter={(v, name) => [fmtNum(v), name === 'revenue' ? 'Revenue' : 'Net Income']}
            />
            <Bar dataKey="revenue" fill={REVENUE_COLOR} opacity={0.8} radius={[3, 3, 0, 0]} />
            <Bar dataKey="net" radius={[3, 3, 0, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.net >= 0 ? NET_COLOR : '#ef4444'} opacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statement Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
          <thead>
            <tr>
              {['Period', 'Revenue', 'Net Income', 'Gross Profit', 'Op. Cash Flow', 'Free Cash Flow', 'Total Debt'].map(h => (
                <th key={h} style={{ color: '#64748b', fontWeight: 600, padding: '0.4rem 0.6rem', textAlign: h === 'Period' ? 'left' : 'right', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {statements.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                <td style={{ color: '#e2e8f0', padding: '0.5rem 0.6rem' }}>{s.report_date?.slice(0, 7)}</td>
                <td style={{ color: '#94a3b8', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.total_revenue)}</td>
                <td style={{ color: s.net_income >= 0 ? '#4ade80' : '#f87171', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.net_income)}</td>
                <td style={{ color: '#94a3b8', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.gross_profit)}</td>
                <td style={{ color: '#94a3b8', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.operating_cash_flow)}</td>
                <td style={{ color: '#94a3b8', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.free_cash_flow)}</td>
                <td style={{ color: '#94a3b8', padding: '0.5rem 0.6rem', textAlign: 'right' }}>{fmtNum(s.total_debt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Corporate Actions */}
      {actions?.length > 0 && (
        <div>
          <p style={{ color: '#94a3b8', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Recent Dividends & Splits
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {actions.map((a, i) => (
              <span key={i} style={{
                fontSize: '0.7rem', padding: '0.2rem 0.6rem', borderRadius: '1rem',
                background: a.action_type === 'dividend' ? 'rgba(34,197,94,0.1)' : 'rgba(99,102,241,0.1)',
                color: a.action_type === 'dividend' ? '#4ade80' : '#a5b4fc',
                border: a.action_type === 'dividend' ? '1px solid rgba(34,197,94,0.2)' : '1px solid rgba(99,102,241,0.2)',
              }}>
                {a.date?.slice(0, 10)} · {a.action_type === 'dividend' ? `÷ $${a.value.toFixed(3)}` : `Split ${a.value}x`}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const spinnerStyle = {
  width: 32, height: 32,
  border: '3px solid rgba(99,102,241,0.15)',
  borderTopColor: '#6366f1',
  borderRadius: '50%',
  animation: 'assetSpin 0.9s linear infinite',
};

// ─── Main AssetModal ──────────────────────────────────────────────────────────
export default function AssetModal({ ticker, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('performance');

  useEffect(() => {
    (async () => {
      setLoading(true); setError(null);
      try {
        const res = await api.get(`/market/asset/${ticker}`);
        setData(res.data);
      } catch (e) {
        setError(e.response?.data?.detail || 'Fundamentals not yet available — data mining in progress.');
      } finally {
        setLoading(false);
      }
    })();
  }, [ticker]);

  const info = data?.info ?? {};
  const statements = data?.statements ?? [];
  const actions = data?.actions ?? [];

  // Build badge sentiment logic
  const badges = [
    { label: 'Market Cap', value: fmtNum(info.market_cap), sentiment: info.market_cap > 1e9 ? 'positive' : 'neutral' },
    { label: 'P/E (Fwd)', value: fmtRaw(info.forward_pe), sentiment: info.forward_pe != null && info.forward_pe < 30 ? 'positive' : info.forward_pe ? 'negative' : 'neutral' },
    { label: 'P/E (Trailing)', value: fmtRaw(info.trailing_pe), sentiment: 'neutral' },
    { label: 'P/B Ratio', value: fmtMultiple(info.price_to_book), sentiment: info.price_to_book != null && info.price_to_book < 3 ? 'positive' : info.price_to_book ? 'negative' : 'neutral' },
    { label: 'Div Yield', value: info.dividend_yield != null ? `${info.dividend_yield.toFixed(2)}%` : '—', sentiment: info.dividend_yield > 1 ? 'positive' : 'neutral' },
    { label: 'Profit Margin', value: fmtPct(info.profit_margins), sentiment: info.profit_margins > 0.1 ? 'positive' : info.profit_margins < 0 ? 'negative' : 'neutral' },
    { label: 'ROE', value: fmtPct(info.return_on_equity), sentiment: info.return_on_equity > 0.15 ? 'positive' : info.return_on_equity < 0 ? 'negative' : 'neutral' },
    { label: 'Beta', value: fmtRaw(info.beta), sentiment: info.beta != null && info.beta < 1.5 ? 'positive' : info.beta ? 'negative' : 'neutral' },
    { label: 'D/E Ratio', value: fmtRaw(info.debt_to_equity), sentiment: info.debt_to_equity != null && info.debt_to_equity < 1 ? 'positive' : info.debt_to_equity ? 'negative' : 'neutral' },
    { label: 'Short Ratio', value: fmtRaw(info.short_ratio), sentiment: info.short_ratio != null && info.short_ratio < 3 ? 'positive' : info.short_ratio ? 'negative' : 'neutral' },
    { label: 'Insider Hold', value: fmtPct(info.held_percent_insiders), sentiment: info.held_percent_insiders > 0.05 ? 'positive' : 'neutral' },
    { label: 'Instit. Hold', value: fmtPct(info.held_percent_institutions), sentiment: info.held_percent_institutions > 0.5 ? 'positive' : 'neutral' },
    ...(info.annual_report_expense_ratio != null ? [{ label: 'Expense Ratio', value: fmtPct(info.annual_report_expense_ratio), sentiment: info.annual_report_expense_ratio < 0.002 ? 'positive' : info.annual_report_expense_ratio > 0.005 ? 'negative' : 'neutral' }] : []),
    ...(info.five_year_average_return != null ? [{ label: '5yr Avg Return', value: fmtPct(info.five_year_average_return), sentiment: info.five_year_average_return > 0.08 ? 'positive' : info.five_year_average_return < 0 ? 'negative' : 'neutral' }] : []),
  ].filter(b => b.value !== '—');

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '1rem',
      }}
    >
      <style>{`@keyframes assetSpin { to { transform: rotate(360deg); } } @keyframes assetFadeIn { from { opacity: 0; transform: translateY(12px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }`}</style>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'linear-gradient(145deg, #0d1526, #111827)',
          border: '1px solid rgba(99,102,241,0.18)',
          borderRadius: '1.25rem',
          width: '100%', maxWidth: 820,
          maxHeight: '90vh',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 30px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04)',
          animation: 'assetFadeIn 0.2s ease-out',
          overflow: 'hidden',
        }}
      >
        {/* ── Header ── */}
        <div style={{
          padding: '1.5rem 1.75rem 1.25rem',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <h2 style={{ color: '#f1f5f9', fontSize: '1.5rem', fontWeight: 800, margin: 0, letterSpacing: '-0.03em' }}>
                  {ticker}
                </h2>
                {info.asset_type && (
                  <span style={{
                    fontSize: '0.65rem', fontWeight: 700, padding: '0.2rem 0.55rem',
                    borderRadius: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.08em',
                    background: info.asset_type === 'ETF' ? 'rgba(245,158,11,0.15)' : 'rgba(99,102,241,0.15)',
                    color: info.asset_type === 'ETF' ? '#fbbf24' : '#a5b4fc',
                    border: info.asset_type === 'ETF' ? '1px solid rgba(245,158,11,0.25)' : '1px solid rgba(99,102,241,0.25)',
                  }}>
                    {info.asset_type}
                  </span>
                )}
              </div>
              {info.name && (
                <p style={{ color: '#64748b', fontSize: '0.82rem', margin: '0.25rem 0 0' }}>{info.name}</p>
              )}
              {(info.sector || info.industry) && (
                <p style={{ color: '#475569', fontSize: '0.72rem', margin: '0.15rem 0 0' }}>
                  {[info.sector, info.industry].filter(Boolean).join(' · ')}
                </p>
              )}
            </div>
            <button onClick={onClose} style={{
              background: 'rgba(255,255,255,0.06)', border: 'none', color: '#94a3b8',
              width: 32, height: 32, borderRadius: '50%', cursor: 'pointer',
              fontSize: '1.1rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0, transition: 'background 0.15s',
            }}
              onMouseOver={e => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
              onMouseOut={e => e.currentTarget.style.background = 'rgba(255,255,255,0.06)'}
            >
              ×
            </button>
          </div>

          {/* Description */}
          {info.description && (
            <p style={{
              color: '#94a3b8', fontSize: '0.78rem', lineHeight: 1.6,
              marginTop: '0.9rem', maxHeight: '3.5rem', overflow: 'hidden',
              display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical',
            }}>
              {info.description}
            </p>
          )}
        </div>

        {/* Scrollable body */}
        <div style={{ overflowY: 'auto', flex: 1, padding: '1.25rem 1.75rem 1.5rem' }}>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 280, gap: '1rem' }}>
              <div style={spinnerStyle} />
              <p style={{ color: '#475569', fontSize: '0.82rem' }}>Loading fundamentals...</p>
            </div>
          ) : error ? (
            <div style={{ textAlign: 'center', minHeight: 280, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <p style={{ fontSize: '1.5rem' }}>📡</p>
              <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: 380, lineHeight: 1.6 }}>{error}</p>
            </div>
          ) : (
            <>
              {/* Metric Badges */}
              {badges.length > 0 && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                  gap: '0.5rem',
                  marginBottom: '1.5rem',
                }}>
                  {badges.map((b, i) => <Badge key={i} {...b} />)}
                </div>
              )}

              {/* Target Price Callout */}
              {info.target_mean_price != null && (
                <div style={{
                  marginBottom: '1.25rem', padding: '0.75rem 1rem',
                  background: 'rgba(99,102,241,0.08)', borderRadius: '0.6rem',
                  border: '1px solid rgba(99,102,241,0.2)',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Analyst Consensus Target</span>
                  <span style={{ color: '#a5b4fc', fontWeight: 700, fontSize: '1rem' }}>${info.target_mean_price.toFixed(2)}</span>
                </div>
              )}

              {/* Tab Navigation */}
              <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem' }}>
                <TabBtn label="Performance" active={tab === 'performance'} onClick={() => setTab('performance')} />
                <TabBtn label="Financials" active={tab === 'financials'} onClick={() => setTab('financials')} />
              </div>

              {/* Tab Content */}
              {tab === 'performance'
                ? <HistoryTab ticker={ticker} />
                : <FinancialsTab statements={statements} actions={actions} />
              }
            </>
          )}
        </div>
      </div>
    </div>
  );
}
