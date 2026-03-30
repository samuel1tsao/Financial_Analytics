import { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import useStore from '../../store';
import api from '../../api/client';

// ─── Formatters ──────────────────────────────────────────────────────────────
const fmtMoney = (val) => {
  if (val >= 1_000_000_000) return `$${(val / 1_000_000_000).toFixed(1)}B`;
  if (val >= 1_000_000) return `$${(val / 1_000_000).toFixed(1)}M`;
  if (val >= 1_000) return `$${(val / 1_000).toFixed(0)}K`;
  return `$${val?.toFixed(0) || 0}`;
};

function EquityTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{
      background: 'rgba(15,23,42,0.95)', backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.1)', borderRadius: '0.6rem',
      padding: '0.75rem 1rem', boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
    }}>
      <p style={{ color: '#94a3b8', fontSize: '0.75rem', marginBottom: '0.4rem' }}>
        {d.date}
      </p>
      <p style={{ color: '#4ade80', fontWeight: 600, fontSize: '0.9rem' }}>
        Value: {fmtMoney(d.value)}
      </p>
    </div>
  );
}

const partyColor = {
  Democrat: '#3b82f6',
  Republican: '#ef4444',
};

const partyBg = {
  Democrat: 'rgba(59,130,246,0.1)',
  Republican: 'rgba(239,68,68,0.1)',
};

// ─── Official Card ───────────────────────────────────────────────────────────
function OfficialCard({ official, onMimic, onFavorite, isFavorited }) {
  const [mimicking, setMimicking] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const topHoldings = Object.entries(official.portfolio)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  const handleMimic = async (e) => {
    e.stopPropagation();
    setMimicking(true);
    await onMimic(official.id);
    setMimicking(false);
  };

  return (
    <div
      style={{
        borderRadius: '0.85rem',
        background: 'rgba(15, 23, 42, 0.7)',
        backdropFilter: 'blur(16px)',
        border: '1px solid rgba(255,255,255,0.06)',
        overflow: 'hidden',
        transition: 'border-color 0.25s, transform 0.25s, box-shadow 0.25s',
        cursor: 'pointer',
      }}
      onMouseOver={(e) => {
        e.currentTarget.style.borderColor = 'rgba(59,130,246,0.25)';
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.boxShadow = '0 12px 40px rgba(0,0,0,0.3)';
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header */}
      <div style={{ padding: '1.25rem 1.25rem 0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
              <h3 style={{
                color: '#f1f5f9', fontWeight: 700, fontSize: '1.05rem',
                margin: 0, letterSpacing: '-0.01em',
              }}>
                {official.name}
              </h3>
              <span style={{
                fontSize: '0.6rem', fontWeight: 700,
                padding: '0.15rem 0.5rem', borderRadius: '1rem',
                background: partyBg[official.party] || 'rgba(100,100,100,0.2)',
                color: partyColor[official.party] || '#94a3b8',
                letterSpacing: '0.04em',
              }}>
                {official.party?.[0]}
              </span>
            </div>
            <p style={{ color: '#64748b', fontSize: '0.75rem', margin: 0 }}>
              {official.title} · {official.state}
            </p>
          </div>
          {/* Favorite star */}
          <button
            onClick={(e) => { e.stopPropagation(); onFavorite(official.id); }}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: '1.2rem', padding: '0.2rem',
              transition: 'transform 0.2s',
              transform: isFavorited ? 'scale(1.15)' : 'scale(1)',
            }}
            title={isFavorited ? 'Remove from favorites' : 'Add to favorites'}
          >
            {isFavorited ? '⭐' : '☆'}
          </button>
        </div>

        {/* Value + Performance Row */}
        <div style={{
          display: 'flex', gap: '1.25rem', marginTop: '0.85rem',
          alignItems: 'baseline',
        }}>
          <div>
            <p style={{ color: '#475569', fontSize: '0.65rem', fontWeight: 500, margin: 0 }}>
              PORTFOLIO VALUE
            </p>
            <p style={{
              color: '#e2e8f0', fontWeight: 700, fontSize: '1.2rem',
              margin: '0.15rem 0 0', letterSpacing: '-0.02em',
            }}>
              {fmtMoney(official.total_value)}
            </p>
          </div>
          {official.performance_1y != null && (
            <div>
              <p style={{ color: '#475569', fontSize: '0.65rem', fontWeight: 500, margin: 0 }}>1Y</p>
              <p style={{
                color: official.performance_1y >= 0 ? '#4ade80' : '#f87171',
                fontWeight: 600, fontSize: '0.95rem', margin: '0.15rem 0 0',
              }}>
                {official.performance_1y >= 0 ? '+' : ''}{(official.performance_1y * 100).toFixed(1)}%
              </p>
            </div>
          )}
          {official.performance_5y != null && (
            <div>
              <p style={{ color: '#475569', fontSize: '0.65rem', fontWeight: 500, margin: 0 }}>5Y</p>
              <p style={{
                color: official.performance_5y >= 0 ? '#4ade80' : '#f87171',
                fontWeight: 600, fontSize: '0.95rem', margin: '0.15rem 0 0',
              }}>
                {official.performance_5y >= 0 ? '+' : ''}{(official.performance_5y * 100).toFixed(1)}%
              </p>
            </div>
          )}
          {/* Data source badge */}
          {official.data_source && (
            <div style={{ marginLeft: 'auto' }}>
              <span style={{
                fontSize: '0.55rem', fontWeight: 600,
                padding: '0.15rem 0.45rem', borderRadius: '0.3rem',
                background: official.data_source === 'database'
                  ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
                border: official.data_source === 'database'
                  ? '1px solid rgba(34,197,94,0.2)' : '1px solid rgba(245,158,11,0.2)',
                color: official.data_source === 'database' ? '#4ade80' : '#fbbf24',
                letterSpacing: '0.04em',
              }}>
                {official.data_source === 'database' ? '● LIVE' : '● CURATED'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Top Holdings Bar */}
      <div style={{ padding: '0 1.25rem', marginTop: '0.6rem' }}>
        <p style={{ color: '#475569', fontSize: '0.6rem', fontWeight: 600, marginBottom: '0.3rem', letterSpacing: '0.06em' }}>
          TOP HOLDINGS
        </p>
        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
          {topHoldings.map(([ticker, weight]) => (
            <span key={ticker} style={{
              fontSize: '0.7rem', fontWeight: 600,
              padding: '0.2rem 0.5rem', borderRadius: '0.3rem',
              background: 'rgba(59,130,246,0.08)',
              border: '1px solid rgba(59,130,246,0.15)',
              color: '#93c5fd',
            }}>
              {ticker} {(weight * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      </div>

      {/* Expanded: Recent Trades */}
      {expanded && (
        <div style={{
          padding: '0.75rem 1.25rem 0',
          marginTop: '0.6rem',
          borderTop: '1px solid rgba(255,255,255,0.04)',
          animation: 'fadeIn 0.2s ease',
        }}>
          <p style={{ color: '#475569', fontSize: '0.6rem', fontWeight: 600, marginBottom: '0.4rem', letterSpacing: '0.06em' }}>
            HISTORICAL PERFORMANCE
          </p>
          {official.historical_equity && official.historical_equity.length > 0 ? (
            <div style={{ height: 180, width: '100%', marginBottom: '1.5rem' }}>
              <ResponsiveContainer>
                <AreaChart data={official.historical_equity} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gradPerf" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4ade80" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#4ade80" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                  <XAxis dataKey="date" stroke="#475569" fontSize={10} tickLine={false} tickFormatter={(tick) => {
                    const d = new Date(tick);
                    return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`;
                  }} />
                  <YAxis stroke="#475569" fontSize={10} tickLine={false} tickFormatter={fmtMoney} width={50} />
                  <Tooltip content={<EquityTooltip />} />
                  <Area type="monotone" dataKey="value" stroke="#4ade80" strokeWidth={2} fill="url(#gradPerf)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ padding: '1.5rem 0', textAlign: 'center', color: '#64748b', fontSize: '0.8rem', fontStyle: 'italic', marginBottom: '1rem' }}>
              Not enough historical data to plot equity curve.
            </div>
          )}

          <p style={{ color: '#475569', fontSize: '0.6rem', fontWeight: 600, marginBottom: '0.4rem', letterSpacing: '0.06em' }}>
            RECENT TRADES
          </p>
          {official.top_trades?.map((trade, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              marginBottom: '0.3rem', fontSize: '0.78rem',
            }}>
              <span style={{
                width: 38, textAlign: 'center',
                fontSize: '0.6rem', fontWeight: 700,
                padding: '0.15rem 0', borderRadius: '0.25rem',
                background: trade.action === 'BUY' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                color: trade.action === 'BUY' ? '#4ade80' : '#f87171',
              }}>
                {trade.action}
              </span>
              <span style={{ color: '#e2e8f0', fontWeight: 600, width: 55 }}>{trade.ticker}</span>
              <span style={{ color: '#94a3b8' }}>{fmtMoney(trade.amount)}</span>
              <span style={{ color: '#475569', marginLeft: 'auto', fontSize: '0.7rem' }}>{trade.date}</span>
            </div>
          ))}

          {/* Full Allocation */}
          <p style={{ color: '#475569', fontSize: '0.6rem', fontWeight: 600, marginTop: '0.75rem', marginBottom: '0.35rem', letterSpacing: '0.06em' }}>
            FULL ALLOCATION
          </p>
          <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: '0.15rem' }}>
            {Object.entries(official.portfolio)
              .sort(([, a], [, b]) => b - a)
              .map(([ticker, weight], idx) => {
                const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#22c55e', '#ec4899', '#f97316', '#6366f1', '#14b8a6', '#a855f7', '#64748b'];
                return (
                  <div
                    key={ticker}
                    title={`${ticker}: ${(weight * 100).toFixed(1)}%`}
                    style={{
                      width: `${weight * 100}%`,
                      background: colors[idx % colors.length],
                      transition: 'width 0.3s',
                    }}
                  />
                );
              })}
          </div>
          <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginBottom: '0.25rem' }}>
            {Object.entries(official.portfolio)
              .sort(([, a], [, b]) => b - a)
              .map(([ticker, weight], idx) => {
                const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#22c55e', '#ec4899', '#f97316', '#6366f1', '#14b8a6', '#a855f7', '#64748b'];
                return (
                  <span key={ticker} style={{ fontSize: '0.6rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                    <span style={{ width: 6, height: 6, borderRadius: 2, background: colors[idx % colors.length], display: 'inline-block' }} />
                    {ticker} {(weight * 100).toFixed(0)}%
                  </span>
                );
              })}
          </div>
        </div>
      )}

      {/* Action Footer */}
      <div style={{
        display: 'flex', gap: '0.5rem',
        padding: '0.75rem 1.25rem',
        marginTop: '0.5rem',
        borderTop: '1px solid rgba(255,255,255,0.04)',
      }}>
        <button
          onClick={handleMimic}
          disabled={mimicking}
          style={{
            flex: 1, padding: '0.5rem',
            borderRadius: '0.45rem',
            background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))',
            border: '1px solid rgba(59,130,246,0.25)',
            color: '#60a5fa', fontWeight: 600, fontSize: '0.78rem',
            cursor: 'pointer',
            transition: 'all 0.2s',
            opacity: mimicking ? 0.6 : 1,
          }}
          onMouseOver={(e) => {
            if (!mimicking) {
              e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.25))';
            }
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))';
          }}
        >
          {mimicking ? 'Copying...' : '📋 Mimic Portfolio'}
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onFavorite(official.id); }}
          style={{
            padding: '0.5rem 0.85rem',
            borderRadius: '0.45rem',
            background: isFavorited ? 'rgba(245,158,11,0.12)' : 'rgba(255,255,255,0.04)',
            border: isFavorited
              ? '1px solid rgba(245,158,11,0.3)'
              : '1px solid rgba(255,255,255,0.08)',
            color: isFavorited ? '#fbbf24' : '#64748b',
            fontWeight: 600, fontSize: '0.78rem',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {isFavorited ? '★ Tracked' : '☆ Track'}
        </button>
      </div>
    </div>
  );
}

// ─── Search & Filter Bar ─────────────────────────────────────────────────────
function FilterBar({ search, setSearch, filter, setFilter, chamberFilter, setChamberFilter, sortBy, setSortBy }) {
  const filterBtnStyle = (active) => ({
    padding: '0.4rem 0.85rem', borderRadius: '2rem',
    fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer',
    border: active ? '1px solid rgba(59,130,246,0.3)' : '1px solid rgba(255,255,255,0.08)',
    background: active ? 'rgba(59,130,246,0.12)' : 'rgba(255,255,255,0.03)',
    color: active ? '#60a5fa' : '#94a3b8',
    transition: 'all 0.15s',
  });

  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '0.5rem',
        alignItems: 'center', marginBottom: '0.5rem',
      }}>
        <input
          type="text"
          placeholder="Search officials..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: '0.55rem 1rem', borderRadius: '0.5rem',
            background: 'rgba(30,41,59,0.8)',
            border: '1px solid rgba(255,255,255,0.08)',
            color: '#f1f5f9', fontSize: '0.85rem',
            outline: 'none', width: 220,
            transition: 'border-color 0.2s',
          }}
          onFocus={(e) => e.target.style.borderColor = 'rgba(59,130,246,0.4)'}
          onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
        />

        {/* Party filters */}
        <button style={filterBtnStyle(filter === 'all')} onClick={() => setFilter('all')}>All</button>
        <button style={filterBtnStyle(filter === 'Democrat')} onClick={() => setFilter('Democrat')}>
          <span style={{ color: '#3b82f6' }}>●</span> Democrat
        </button>
        <button style={filterBtnStyle(filter === 'Republican')} onClick={() => setFilter('Republican')}>
          <span style={{ color: '#ef4444' }}>●</span> Republican
        </button>

        {/* Separator */}
        <span style={{ color: 'rgba(255,255,255,0.1)', margin: '0 0.15rem' }}>|</span>

        {/* Chamber filters */}
        <button style={filterBtnStyle(chamberFilter === 'all')} onClick={() => setChamberFilter('all')}>All Chambers</button>
        <button style={filterBtnStyle(chamberFilter === 'house')} onClick={() => setChamberFilter('house')}>🏛️ House</button>
        <button style={filterBtnStyle(chamberFilter === 'senate')} onClick={() => setChamberFilter('senate')}>🏛️ Senate</button>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span style={{ color: '#475569', fontSize: '0.7rem' }}>Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              padding: '0.35rem 0.6rem', borderRadius: '0.4rem',
              background: 'rgba(30,41,59,0.8)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#94a3b8', fontSize: '0.75rem', outline: 'none',
              cursor: 'pointer',
            }}
          >
            <option value="value">Portfolio Value</option>
            <option value="perf1y">1Y Performance</option>
            <option value="perf5y">5Y Performance</option>
            <option value="name">Name</option>
          </select>
        </div>
      </div>
    </div>
  );
}

// ─── Main Directory ──────────────────────────────────────────────────────────
export default function Directory() {
  const { favorites, fetchUserData } = useStore();
  const [officials, setOfficials] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [chamberFilter, setChamberFilter] = useState('all');
  const [sortBy, setSortBy] = useState('value');
  const [mimicSuccess, setMimicSuccess] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/officials');
        setOfficials(res.data.officials);
      } catch (err) {
        console.error('Failed to load officials:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleMimic = async (officialId) => {
    try {
      const res = await api.post(`/officials/${officialId}/mimic`);
      setMimicSuccess(res.data.profile_name);
      await fetchUserData();
      setTimeout(() => setMimicSuccess(''), 3000);
    } catch (err) {
      console.error('Mimic failed:', err);
    }
  };

  const handleFavorite = async (officialId) => {
    try {
      const token = localStorage.getItem('token');
      await fetch(`http://localhost:8000/api/v1/user/favorites?official_id=${officialId}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      await fetchUserData();
    } catch (err) {
      console.error('Favorite toggle failed:', err);
    }
  };

  // ─── Filter + Sort ───────────────────────────────────────────────────────
  let filtered = officials.filter((o) => {
    const matchSearch = o.name.toLowerCase().includes(search.toLowerCase())
      || (o.state || '').toLowerCase().includes(search.toLowerCase())
      || (o.title || '').toLowerCase().includes(search.toLowerCase());
    const matchParty = filter === 'all' || o.party === filter;
    const matchChamber = chamberFilter === 'all'
      || (chamberFilter === 'house' && (o.title || '').toLowerCase().includes('representative'))
      || (chamberFilter === 'house' && (o.title || '').toLowerCase().includes('speaker'))
      || (chamberFilter === 'senate' && (o.title || '').toLowerCase().includes('senator'))
      || (chamberFilter === 'senate' && (o.title || '').toLowerCase().includes('senate'));
    return matchSearch && matchParty && matchChamber;
  });

  filtered.sort((a, b) => {
    switch (sortBy) {
      case 'value': return (b.total_value || 0) - (a.total_value || 0);
      case 'perf1y': return (b.performance_1y || 0) - (a.performance_1y || 0);
      case 'perf5y': return (b.performance_5y || 0) - (a.performance_5y || 0);
      case 'name': return a.name.localeCompare(b.name);
      default: return 0;
    }
  });

  // ─── Render ──────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#64748b',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: 40, height: 40, margin: '0 auto 1rem',
            borderRadius: '50%', border: '3px solid rgba(59,130,246,0.2)',
            borderTopColor: '#3b82f6',
            animation: 'spin 1s linear infinite',
          }} />
          <p>Loading officials data...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, padding: '2rem', overflow: 'auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h1 style={{
          color: '#f1f5f9', fontWeight: 700, fontSize: '1.5rem',
          margin: 0, letterSpacing: '-0.02em',
        }}>
          Public Officials Directory
        </h1>
        <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.35rem' }}>
          Browse disclosed portfolios of U.S. lawmakers. Mimic their holdings or track their trades.
        </p>
      </div>

      {/* Mimic Success Toast */}
      {mimicSuccess && (
        <div style={{
          padding: '0.65rem 1rem', borderRadius: '0.5rem',
          background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)',
          color: '#4ade80', fontSize: '0.85rem', marginBottom: '1rem',
          display: 'flex', alignItems: 'center', gap: '0.5rem',
          animation: 'fadeIn 0.3s ease',
        }}>
          <span>✅</span>
          <span>Created "<strong>{mimicSuccess}</strong>" — check your sidebar!</span>
        </div>
      )}

      <FilterBar
        search={search} setSearch={setSearch}
        filter={filter} setFilter={setFilter}
        chamberFilter={chamberFilter} setChamberFilter={setChamberFilter}
        sortBy={sortBy} setSortBy={setSortBy}
      />

      {/* Card Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
        gap: '1rem',
      }}>
        {filtered.map((official) => (
          <OfficialCard
            key={official.id}
            official={official}
            onMimic={handleMimic}
            onFavorite={handleFavorite}
            isFavorited={favorites.includes(official.id)}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '3rem', color: '#475569',
        }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🔍</p>
          <p>No officials match your search.</p>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
