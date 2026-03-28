import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import useStore from '../../store';
import api from '../../api/client';

// ─── Formatters ──────────────────────────────────────────────────────────────
const fmt = (val) => {
  if (val >= 1_000_000) return `$${(val / 1_000_000).toFixed(1)}M`;
  if (val >= 1_000) return `$${(val / 1_000).toFixed(0)}K`;
  return `$${val.toFixed(0)}`;
};

const pct = (v) => `${(v * 100).toFixed(1)}%`;

// ─── Custom Tooltip ──────────────────────────────────────────────────────────
function SimTooltip({ active, payload, label }) {
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
        Year {d.year}
      </p>
      <p style={{ color: '#60a5fa', fontWeight: 600, fontSize: '0.9rem' }}>
        Expected: {fmt(d.expected)}
      </p>
      <p style={{ color: 'rgba(139,92,246,0.8)', fontSize: '0.8rem' }}>
        Best Case: {fmt(d.upper)}
      </p>
      <p style={{ color: 'rgba(239,68,68,0.7)', fontSize: '0.8rem' }}>
        Worst Case: {fmt(d.lower)}
      </p>
    </div>
  );
}

// ─── Stat Card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, subtext, color = '#60a5fa' }) {
  return (
    <div style={{
      padding: '1.25rem', borderRadius: '0.75rem',
      background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.06)',
      transition: 'border-color 0.2s, transform 0.2s',
    }}
      onMouseOver={(e) => {
        e.currentTarget.style.borderColor = 'rgba(59,130,246,0.3)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      <p style={{ color: '#64748b', fontSize: '0.75rem', fontWeight: 500, marginBottom: '0.3rem' }}>
        {label}
      </p>
      <p style={{ color, fontWeight: 700, fontSize: '1.4rem', letterSpacing: '-0.02em' }}>
        {value}
      </p>
      {subtext && (
        <p style={{ color: '#475569', fontSize: '0.7rem', marginTop: '0.2rem' }}>{subtext}</p>
      )}
    </div>
  );
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────
export default function DashboardMain() {
  const {
    portfolios, activePortfolioId, hasCompletedQuestionnaire, fetchUserData,
  } = useStore();
  const navigate = useNavigate();

  const [simulation, setSimulation] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState('');

  const activePortfolio = portfolios.find((p) => p.id === activePortfolioId)
    || portfolios.find((p) => p.is_current)
    || null;

  // Fetch simulation whenever the active portfolio changes
  useEffect(() => {
    if (!activePortfolio) return;
    let cancelled = false;
    (async () => {
      setSimLoading(true);
      setSimError('');
      try {
        const res = await api.post('/simulate', null, {
          params: {
            portfolio_id: activePortfolio.id,
            initial_investment: 100000,
            projection_years: 30,
          },
        });
        if (!cancelled) setSimulation(res.data.simulation);
      } catch (err) {
        if (!cancelled) setSimError(err.response?.data?.detail || 'Simulation failed');
      } finally {
        if (!cancelled) setSimLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [activePortfolio?.id]);

  // ─── Empty State ─────────────────────────────────────────────────────────
  if (!hasCompletedQuestionnaire && portfolios.length === 0) {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '3rem', textAlign: 'center',
      }}>
        <div style={{
          width: 80, height: 80, borderRadius: '50%',
          background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '2rem', marginBottom: '1.5rem',
          border: '1px solid rgba(59,130,246,0.2)',
        }}>
          📊
        </div>
        <h2 style={{
          color: '#f1f5f9', fontWeight: 700, fontSize: '1.5rem', marginBottom: '0.75rem',
        }}>
          Welcome to Stock Recommender
        </h2>
        <p style={{
          color: '#94a3b8', maxWidth: 460, lineHeight: 1.6, marginBottom: '2rem',
        }}>
          Get started by completing your investment questionnaire. We'll analyze your risk profile,
          financial goals, and preferences to build a personalized portfolio recommendation.
        </p>
        <button
          id="start-questionnaire-btn"
          onClick={() => navigate('/questionnaire')}
          style={{
            padding: '0.85rem 2rem', borderRadius: '0.6rem',
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            color: '#fff', fontWeight: 600, fontSize: '0.95rem',
            border: 'none', cursor: 'pointer',
            boxShadow: '0 4px 20px rgba(59,130,246,0.3)',
            transition: 'transform 0.2s, box-shadow 0.2s',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 8px 30px rgba(59,130,246,0.4)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 20px rgba(59,130,246,0.3)';
          }}
        >
          Start Questionnaire
        </button>
        <button
          onClick={() => navigate('/questionnaire')}
          style={{
            marginTop: '1rem', background: 'none', border: 'none',
            color: '#64748b', cursor: 'pointer', fontSize: '0.8rem',
          }}
        >
          Skip for now →
        </button>
      </div>
    );
  }

  // ─── Build chart data ──────────────────────────────────────────────────
  const chartData = simulation ? simulation.years.map((yr, i) => ({
    year: yr,
    expected: simulation.expected_path[i],
    upper: simulation.upper_bound[i],
    lower: simulation.lower_bound[i],
  })) : [];

  const stats = simulation?.portfolio_stats || {};

  // ─── Active Portfolio View ─────────────────────────────────────────────
  return (
    <div style={{ flex: 1, padding: '2rem', overflow: 'auto' }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: '1.5rem',
      }}>
        <div>
          <h1 style={{
            color: '#f1f5f9', fontWeight: 700, fontSize: '1.5rem',
            margin: 0, letterSpacing: '-0.02em',
          }}>
            {activePortfolio ? activePortfolio.profile_name : 'Dashboard'}
          </h1>
          {activePortfolio?.is_current && (
            <span style={{
              fontSize: '0.7rem', background: 'rgba(34,197,94,0.12)',
              color: '#4ade80', padding: '0.2rem 0.6rem',
              borderRadius: '1rem', fontWeight: 600,
              display: 'inline-block', marginTop: '0.3rem',
            }}>
              ACTIVE PORTFOLIO
            </span>
          )}
        </div>
        <button
          onClick={() => navigate('/questionnaire')}
          style={{
            padding: '0.5rem 1rem', borderRadius: '0.5rem',
            background: 'rgba(59,130,246,0.1)',
            border: '1px solid rgba(59,130,246,0.2)',
            color: '#60a5fa', cursor: 'pointer',
            fontWeight: 500, fontSize: '0.8rem',
            transition: 'all 0.2s',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'rgba(59,130,246,0.2)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'rgba(59,130,246,0.1)';
          }}
        >
          Modify Preferences
        </button>
      </div>

      {/* Stats Summary Row */}
      {simulation && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.75rem', marginBottom: '1.5rem',
        }}>
          <StatCard
            label="Expected Return"
            value={`${stats.expected_annual_return}%`}
            subtext="Annual average"
            color="#4ade80"
          />
          <StatCard
            label="Volatility"
            value={`${stats.annual_volatility}%`}
            subtext="Annual std dev"
            color="#facc15"
          />
          <StatCard
            label="Sharpe Ratio"
            value={stats.sharpe_ratio?.toFixed(2)}
            subtext="Risk-adjusted return"
            color="#60a5fa"
          />
          <StatCard
            label="Projected (30yr)"
            value={fmt(stats.projected_final_expected || 0)}
            subtext={`Range: ${fmt(stats.projected_final_lower || 0)} — ${fmt(stats.projected_final_upper || 0)}`}
            color="#c084fc"
          />
        </div>
      )}

      {/* Simulation Chart */}
      {simLoading && (
        <div style={{
          padding: '3rem', textAlign: 'center', color: '#64748b',
        }}>
          <div style={{
            width: 40, height: 40, margin: '0 auto 1rem',
            borderRadius: '50%', border: '3px solid rgba(59,130,246,0.2)',
            borderTopColor: '#3b82f6',
            animation: 'spin 1s linear infinite',
          }} />
          <p>Running simulation...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {simError && (
        <div style={{
          padding: '1rem', borderRadius: '0.5rem',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          color: '#fca5a5', fontSize: '0.85rem', marginBottom: '1rem',
        }}>
          {simError}
        </div>
      )}

      {simulation && !simLoading && (
        <div style={{
          padding: '1.5rem', borderRadius: '0.75rem',
          background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.06)',
          marginBottom: '1.5rem',
        }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '1rem',
          }}>
            <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem', margin: 0 }}>
              Portfolio Growth Projection
            </h3>
            <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
              ±2σ Confidence Interval · $100K Initial
            </span>
          </div>

          <ResponsiveContainer width="100%" height={380}>
            <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
              <defs>
                <linearGradient id="gradUpper" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradExpected" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="gradLower" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="year"
                stroke="#475569" fontSize={12} tickLine={false}
                label={{ value: 'Years', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }}
              />
              <YAxis
                stroke="#475569" fontSize={11} tickLine={false}
                tickFormatter={fmt}
                width={65}
              />
              <Tooltip content={<SimTooltip />} />

              {/* Confidence band: upper */}
              <Area
                type="monotone" dataKey="upper" name="Best Case (+2σ)"
                stroke="rgba(139,92,246,0.5)" strokeWidth={1.5}
                fill="url(#gradUpper)" dot={false}
                strokeDasharray="4 2"
              />

              {/* Expected path */}
              <Area
                type="monotone" dataKey="expected" name="Expected Path"
                stroke="#3b82f6" strokeWidth={2.5}
                fill="url(#gradExpected)" dot={false}
              />

              {/* Confidence band: lower */}
              <Area
                type="monotone" dataKey="lower" name="Worst Case (−2σ)"
                stroke="rgba(239,68,68,0.4)" strokeWidth={1.5}
                fill="url(#gradLower)" dot={false}
                strokeDasharray="4 2"
              />

              {/* Goal annotations */}
              {simulation.goal_annotations?.map((g, i) => (
                <ReferenceLine
                  key={i}
                  x={g.year}
                  stroke={g.is_short_term ? '#f59e0b' : '#4ade80'}
                  strokeDasharray="3 3"
                  strokeWidth={1.5}
                  label={{
                    value: `${g.label} (Yr ${g.year})`,
                    position: 'top',
                    fill: g.is_short_term ? '#f59e0b' : '#4ade80',
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                />
              ))}

              <Legend
                verticalAlign="top" align="right" height={36}
                wrapperStyle={{ fontSize: '0.75rem', color: '#94a3b8' }}
              />
            </AreaChart>
          </ResponsiveContainer>

          {/* Cash-out events */}
          {simulation.cash_out_events?.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <p style={{ color: '#64748b', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                CASH-OUT EVENTS
              </p>
              {simulation.cash_out_events.map((ev, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: '0.75rem',
                  padding: '0.5rem 0.75rem', borderRadius: '0.4rem',
                  background: 'rgba(245,158,11,0.08)',
                  border: '1px solid rgba(245,158,11,0.15)',
                  marginBottom: '0.35rem',
                }}>
                  <span style={{ color: '#f59e0b', fontSize: '0.85rem' }}>⚡</span>
                  <span style={{ color: '#e2e8f0', fontSize: '0.8rem', flex: 1 }}>
                    <strong>Year {ev.year}</strong> — {ev.goal_name}: withdrew {fmt(ev.amount)}
                  </span>
                  <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
                    Remaining: {fmt(ev.remaining_expected)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Portfolio Allocation Cards */}
      {activePortfolio ? (
        <>
          <h3 style={{
            color: '#e2e8f0', fontWeight: 600, fontSize: '1rem',
            marginBottom: '0.75rem',
          }}>
            Allocation Breakdown
          </h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '0.75rem',
          }}>
            {Object.entries(activePortfolio.weights)
              .sort(([, a], [, b]) => b - a)
              .map(([ticker, weight]) => (
                <div
                  key={ticker}
                  style={{
                    padding: '1rem 1.25rem', borderRadius: '0.75rem',
                    background: 'rgba(15, 23, 42, 0.6)',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    transition: 'border-color 0.2s, transform 0.2s',
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(59,130,246,0.3)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '1.05rem' }}>
                      {ticker}
                    </span>
                    <span style={{ color: '#60a5fa', fontWeight: 600, fontSize: '1.05rem' }}>
                      {pct(weight)}
                    </span>
                  </div>
                  <div style={{
                    marginTop: '0.6rem', height: 5, borderRadius: 3,
                    background: 'rgba(255,255,255,0.06)', overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%', width: `${weight * 100}%`,
                      borderRadius: 3,
                      background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                      transition: 'width 0.5s ease',
                    }} />
                  </div>
                </div>
              ))}
          </div>
        </>
      ) : (
        <p style={{ color: '#64748b' }}>Select a portfolio from the sidebar to view details.</p>
      )}
    </div>
  );
}
