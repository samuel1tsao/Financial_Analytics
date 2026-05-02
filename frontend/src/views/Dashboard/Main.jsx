import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import useStore from '../../store';
import api from '../../api/client';
import AssetModal from '../../components/AssetModal';
import BalanceStepTable from '../../components/BalanceStepTable';

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
    deletePortfolio, questionnaire, simCache, setSimCache
  } = useStore();
  const navigate = useNavigate();

  const [simulation, setSimulation] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState('');
  const [backtestData, setBacktestData] = useState(null);
  const [viewMode, setViewMode] = useState('projection'); // 'projection' or 'backtest'
  
  // Interactive goals state (initialized from active portfolio below)
  const [customGoals, setCustomGoals] = useState([]);
  const [debouncedGoals, setDebouncedGoals] = useState([]);
  
  // ETF Modal
  const [etfModal, setEtfModal] = useState(null);
  const [activeSegmentIdx, setActiveSegmentIdx] = useState(0);
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);

  const activePortfolio = portfolios.find((p) => p.id === activePortfolioId)
    || portfolios.find((p) => p.is_current)
    || null;

  useEffect(() => {
    if (activePortfolio && activePortfolio.weights && activePortfolio.weights.goals) {
      setCustomGoals(activePortfolio.weights.goals);
      setDebouncedGoals(activePortfolio.weights.goals);
    }
  }, [activePortfolio?.id]);

  // ─── Local Recalculation Logic ─────────────────────────────────────────────
  const calculateLocalProjection = (baseSim, goals, startCapital, monthlyInc) => {
    if (!baseSim || !baseSim.expected_annual_returns) return baseSim;

    const totalYears = baseSim.expected_annual_returns.length;
    const goalMap = {};
    goals.forEach(g => {
      goalMap[g.years] = (goalMap[g.years] || 0) + g.amount;
    });

    const runPath = (annualReturns) => {
      let balance = startCapital;
      let debt = 0;
      const path = [balance];
      const contribAnnual = monthlyInc * 12;

      for (let yr = 1; yr <= totalYears; yr++) {
        const r = annualReturns[yr - 1];
        balance *= (1 + r);
        balance += contribAnnual;

        // Debt compounding (10% APR simple approx)
        debt *= 1.10;

        const needed = goalMap[yr] || 0;
        if (needed > 0) {
          if (balance >= needed) {
            balance -= needed;
          } else {
            debt += (needed - balance);
            balance = 0;
          }
        }
        path.push(Math.max(0, balance - debt));
      }
      return path;
    };

    const expectedPath = runPath(baseSim.expected_annual_returns || []);
    const upperBound = runPath(baseSim.upper_annual_returns || []);
    const lowerBound = runPath(baseSim.lower_annual_returns || []);

    // Update stats (rough approximation for CAGR)
    const finalExp = expectedPath.length > 0 ? expectedPath[expectedPath.length - 1] : 0;
    const totalWithdrawn = goals.reduce((sum, g) => sum + (g.amount || 0), 0);
    const cagrBase = (startCapital > 0 && totalYears > 0) ? (finalExp + totalWithdrawn) / startCapital : 0;
    const cagr = (cagrBase > 0) ? (Math.pow(cagrBase, 1 / totalYears) - 1) * 100 : 0;

    return {
      ...baseSim,
      expected_path: expectedPath,
      upper_bound: upperBound,
      lower_bound: lowerBound,
      portfolio_stats: {
        ...(baseSim.portfolio_stats || {}),
        projected_final_expected: finalExp,
        projected_final_upper: upperBound.length > 0 ? upperBound[upperBound.length - 1] : 0,
        projected_final_lower: lowerBound.length > 0 ? lowerBound[lowerBound.length - 1] : 0,
        expected_annual_return: (isNaN(cagr) || !isFinite(cagr)) ? "0.00" : cagr.toFixed(2),
      }
    };
  };

  let weights = {};
  let segments = [];
  let hardConstraints = new Set();
  let startCap = 100000;
  let monthlyContrib = 500;
  
  if (activePortfolio) {
    const data = activePortfolio.weights;
    
    if (data && data.hard_constraints) {
      data.hard_constraints.forEach(c => hardConstraints.add(c.ticker.toUpperCase()));
    }
    
    // NEW: Extract initial conditions from saved weights
    // If not in the portfolio JSON (legacy), fallback to current questionnaire answers
    if (data && data.start_cap) {
      startCap = data.start_cap;
    } else if (questionnaire && questionnaire.start_cap) {
      startCap = questionnaire.start_cap;
    }

    if (data && data.monthly_contrib) {
      monthlyContrib = data.monthly_contrib;
    } else if (questionnaire && questionnaire.monthly_contrib) {
      monthlyContrib = questionnaire.monthly_contrib;
    }

    if (Array.isArray(data)) {
      segments = data;
      const currentSeg = segments[activeSegmentIdx] || segments[0];
      weights = currentSeg?.weights || {};
    } else if (data && data.segments) {
      segments = data.segments;
      const currentSeg = segments[activeSegmentIdx] || segments[0];
      weights = currentSeg?.weights || {};
    } else {
      weights = data || {};
    }
  }

  // Debounce goal input so we don't spam the API while dragging sliders
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedGoals(customGoals), 500);
    return () => clearTimeout(handler);
  }, [customGoals]);

  // Fetch simulation whenever the active portfolio or debounced goals change
  useEffect(() => {
    if (!activePortfolio) return;
    
    // Determine if we need a backend refresh
    // Structural changes (add/delete) require a new glide-path from backend
    const isStructuralChange = simulation === null || 
                               simulation.error ||
                               !simulation.expected_annual_returns ||
                               debouncedGoals.length !== (simulation.goal_annotations?.length || 0);
    
    if (!isStructuralChange && simulation) {
      return;
    }

    let cancelled = false;
    (async () => {
      const cacheKey = `${activePortfolio.id}_${JSON.stringify(debouncedGoals)}`;
      
      if (simCache[cacheKey]) {
        console.log("Using cached simulation for:", cacheKey);
        setSimulation(simCache[cacheKey]);
        setSimLoading(false);
        return;
      }

      setSimLoading(true);
      setSimError('');
      try {
        const res = await api.post('/simulate', {
          portfolio_id: activePortfolio.id,
          initial_investment: startCap,
          monthly_contrib: monthlyContrib,
          projection_years: 30,
          custom_goals_json: JSON.stringify(debouncedGoals),
        });
        
        try {
          const backRes = await api.post('/backtest', {
            portfolio_id: activePortfolio.id,
            initial_investment: startCap,
            monthly_contrib: monthlyContrib,
          });
          if (!cancelled) setBacktestData(backRes.data);
        } catch (e) {
          console.error("Backtest failed", e);
        }

        if (!cancelled) {
          setSimulation(res.data.simulation);
          setSimCache(cacheKey, res.data.simulation);
        }
      } catch (err) {
        console.error('Simulation error:', err);
        if (!cancelled) setSimError('Failed to load growth projection.');
      } finally {
        if (!cancelled) setSimLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [activePortfolio?.id, debouncedGoals.length]);

  // Apply local recalculation to the simulation data
  const activeSimulation = useMemo(() => {
    return calculateLocalProjection(simulation, customGoals, startCap, monthlyContrib);
  }, [simulation, customGoals, startCap, monthlyContrib]);

  // Sync back local simulation to stats
  const chartData = (activeSimulation && !activeSimulation.error) ? activeSimulation.years.map((yr, i) => ({
    year: yr,
    expected: activeSimulation.expected_path[i],
    upper: activeSimulation.upper_bound[i],
    lower: activeSimulation.lower_bound[i],
  })) : [];

  const stats = activeSimulation?.portfolio_stats || {};

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
  // (Removed, moved to useMemo above)

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
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {activePortfolio && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {isConfirmingDelete ? (
                <>
                  <button
                    onClick={async () => {
                      try {
                        console.log("Custom confirm accepted, calling deletePortfolio...");
                        await deletePortfolio(activePortfolio.id);
                        setIsConfirmingDelete(false);
                      } catch (err) {
                        console.error("Critical error in delete handler:", err);
                      }
                    }}
                    style={{
                      padding: '0.5rem 1rem', borderRadius: '0.5rem',
                      background: '#ef4444', border: '1px solid #ef4444',
                      color: '#fff', cursor: 'pointer',
                      fontWeight: 600, fontSize: '0.8rem',
                    }}
                  >
                    Confirm Delete
                  </button>
                  <button
                    onClick={() => setIsConfirmingDelete(false)}
                    style={{
                      padding: '0.5rem 1rem', borderRadius: '0.5rem',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      color: '#94a3b8', cursor: 'pointer',
                      fontWeight: 500, fontSize: '0.8rem',
                    }}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setIsConfirmingDelete(true)}
                  style={{
                    padding: '0.5rem 1rem', borderRadius: '0.5rem',
                    background: 'rgba(239,68,68,0.05)',
                    border: '1px solid rgba(239,68,68,0.2)',
                    color: '#fca5a5', cursor: 'pointer',
                    fontWeight: 500, fontSize: '0.8rem',
                    transition: 'all 0.2s',
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.background = 'rgba(239,68,68,0.15)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.background = 'rgba(239,68,68,0.05)';
                  }}
                >
                  Delete Portfolio
                </button>
              )}
            </div>
          )}
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
      </div>

      {(simError || (activeSimulation && activeSimulation.error)) && (
        <div style={{
          padding: '1rem', borderRadius: '0.5rem',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          color: '#fca5a5', fontSize: '0.85rem', marginBottom: '1.5rem',
        }}>
          {simError || activeSimulation.error}
        </div>
      )}

      {/* Stats Summary Row */}
      {activeSimulation && !activeSimulation.error && (
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

      {activeSimulation && !activeSimulation.error && !simLoading && (
        <>
          {/* View Toggle */}
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem' }}>
            <button
              onClick={() => setViewMode('projection')}
              style={{
                background: viewMode === 'projection' ? 'rgba(59,130,246,0.15)' : 'transparent',
                border: viewMode === 'projection' ? '1px solid rgba(59,130,246,0.4)' : '1px solid rgba(255,255,255,0.06)',
                color: viewMode === 'projection' ? '#60a5fa' : '#64748b',
                padding: '0.4rem 1.25rem', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                transition: 'all 0.2s'
              }}
            >
              30Y PROJECTION
            </button>
            <button
              onClick={() => setViewMode('backtest')}
              style={{
                background: viewMode === 'backtest' ? 'rgba(139,92,246,0.15)' : 'transparent',
                border: viewMode === 'backtest' ? '1px solid rgba(139,92,246,0.4)' : '1px solid rgba(255,255,255,0.06)',
                color: viewMode === 'backtest' ? '#a78bfa' : '#64748b',
                padding: '0.4rem 1.25rem', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                transition: 'all 0.2s'
              }}
            >
              HISTORICAL BACKTEST
            </button>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
            <div style={{
              flex: 3, minWidth: 500,
              padding: '1.5rem', borderRadius: '0.75rem',
              background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                marginBottom: '1rem',
              }}>
                <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem', margin: 0 }}>
                  {viewMode === 'projection' ? 'Portfolio Growth Projection' : '10-Year Historical Backtest'}
                </h3>
                <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
                  {viewMode === 'projection' ? '±2σ Confidence · 30 Year Horizon' : `Actual Returns · ${backtestData?.stats?.start_date} to Present`}
                </span>
              </div>

              <ResponsiveContainer width="100%" height={380}>
                {viewMode === 'projection' ? (
                  <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradExpected" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="year" stroke="#475569" fontSize={12} tickLine={false} />
                    <YAxis stroke="#475569" fontSize={11} tickLine={false} tickFormatter={fmt} width={65} />
                    <Tooltip content={<SimTooltip />} />
                    <Area type="monotone" dataKey="upper" name="Best Case" stroke="rgba(34,197,94,0.4)" fill="rgba(34,197,94,0.05)" dot={false} strokeDasharray="4 2" />
                    <Area type="monotone" dataKey="expected" name="Expected Path" stroke="#3b82f6" strokeWidth={2.5} fill="url(#gradExpected)" dot={false} />
                    <Area type="monotone" dataKey="lower" name="Worst Case" stroke="rgba(239,68,68,0.4)" fill="rgba(239,68,68,0.05)" dot={false} strokeDasharray="4 2" />
                    {activeSimulation.goal_annotations?.map((g, i) => (
                      <ReferenceLine key={i} x={g.year} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: g.label, position: 'top', fill: '#f59e0b', fontSize: 10 }} />
                    ))}
                  </AreaChart>
                ) : (
                  <AreaChart data={backtestData?.backtest || []} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradBack" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="date" stroke="#475569" fontSize={10} tickLine={false} tickFormatter={(d) => d.split('-')[0]} />
                    <YAxis stroke="#475569" fontSize={11} tickLine={false} tickFormatter={fmt} width={65} />
                    <Tooltip 
                      contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '0.5rem' }}
                      labelStyle={{ color: '#94a3b8' }}
                      formatter={(v) => [fmt(v), "Balance"]}
                    />
                    <Area type="monotone" dataKey="balance" name="Historical Performance" stroke="#8b5cf6" strokeWidth={2.5} fill="url(#gradBack)" dot={false} />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>

            <div style={{
              flex: 1, minWidth: 280,
              padding: '1.5rem', borderRadius: '0.75rem',
              background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.9rem', marginBottom: '1rem' }}>
                Interactive Sandbox
              </h3>
              <p style={{ color: '#94a3b8', fontSize: '0.75rem', marginBottom: '1.5rem' }}>
                Test how big purchases or cash-outs affect your long-term growth by overriding your baseline goals here.
              </p>
              
              {customGoals.map((g, i) => (
                <div key={i} style={{ marginBottom: '1.5rem', paddingBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <input 
                    type="text" 
                    value={g.name} 
                    onChange={e => {
                      const ng = [...customGoals];
                      ng[i].name = e.target.value;
                      setCustomGoals(ng);
                    }}
                    style={{
                      width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                      color: '#f1f5f9', padding: '0.4rem', borderRadius: '0.3rem', fontSize: '0.8rem', marginBottom: '0.8rem',
                    }}
                  />
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                    <label style={{ color: '#64748b', fontSize: '0.7rem' }}>Amount: {fmt(g.amount)}</label>
                  </div>
                  <input 
                    type="range" min="0" max="250000" step="5000"
                    value={g.amount}
                    onChange={e => {
                      const ng = [...customGoals];
                      ng[i].amount = Number(e.target.value);
                      setCustomGoals(ng);
                    }}
                    style={{ width: '100%', cursor: 'pointer', marginBottom: '0.8rem' }}
                  />

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                    <label style={{ color: '#64748b', fontSize: '0.7rem' }}>Year: {g.years}</label>
                  </div>
                  <input 
                    type="range" min="1" max="30" step="1"
                    value={g.years}
                    onChange={e => {
                      const ng = [...customGoals];
                      ng[i].years = Number(e.target.value);
                      setCustomGoals(ng);
                    }}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <button
                    onClick={() => {
                      const ng = customGoals.filter((_, idx) => idx !== i);
                      setCustomGoals(ng);
                    }}
                    style={{
                      marginTop: '0.5rem', background: 'transparent', border: 'none',
                      color: '#ef4444', fontSize: '0.7rem', cursor: 'pointer',
                      padding: 0, textDecoration: 'underline'
                    }}
                  >
                    Remove Goal
                  </button>
                </div>
              ))}

              <button
                onClick={() => {
                  setCustomGoals([...customGoals, { name: 'New Goal', amount: 50000, years: 10 }]);
                }}
                style={{
                  width: '100%', padding: '0.6rem', borderRadius: '0.4rem',
                  background: 'rgba(59,130,246,0.1)', border: '1px dashed rgba(59,130,246,0.4)',
                  color: '#60a5fa', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                  marginBottom: '1.5rem', transition: 'all 0.2s'
                }}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(59,130,246,0.2)'}
                onMouseOut={e => e.currentTarget.style.background = 'rgba(59,130,246,0.1)'}
              >
                + Add Custom Goal
              </button>

              {activeSimulation.cash_out_events?.length > 0 && (
                <div style={{ marginTop: '1rem', background: 'rgba(245,158,11,0.08)', padding: '0.75rem', borderRadius: '0.4rem', border: '1px solid rgba(245,158,11,0.15)' }}>
                  <p style={{ color: '#f59e0b', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                    CASH-OUT EVENT LOG
                  </p>
                  {activeSimulation.cash_out_events.map((ev, i) => (
                    <p key={i} style={{ color: '#e2e8f0', fontSize: '0.75rem', margin: '0.2rem 0' }}>
                      <strong>Yr {ev.year}</strong>: {ev.goal_name} (-{fmt(ev.amount)})
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          <BalanceStepTable 
            steps={activeSimulation.step_balances} 
            segments={segments}
            portfolioName={activePortfolio?.profile_name || 'Portfolio'} 
          />
        </>
      )}

      {/* Portfolio Allocation Cards */}
      {activePortfolio ? (
        <div style={{ marginTop: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem', margin: 0 }}>
              Allocation Breakdown 
              {segments.length > 0 && segments[activeSegmentIdx] && (
                <span style={{ color: '#60a5fa', marginLeft: '0.5rem', fontSize: '0.85rem' }}>
                  (Phase {activeSegmentIdx + 1}: Year {segments[activeSegmentIdx].horizon_years?.[0]}-{segments[activeSegmentIdx].horizon_years?.[1]})
                </span>
              )}
            </h3>
            
            {segments.length > 1 && (
              <div style={{ display: 'flex', gap: '0.4rem', background: 'rgba(15,23,42,0.4)', padding: '0.25rem', borderRadius: '0.5rem' }}>
                {segments.map((seg, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveSegmentIdx(i)}
                    style={{
                      padding: '0.3rem 0.7rem', borderRadius: '0.4rem', fontSize: '0.7rem', fontWeight: 600,
                      cursor: 'pointer', border: 'none', transition: 'all 0.2s',
                      background: activeSegmentIdx === i ? '#3b82f6' : 'transparent',
                      color: activeSegmentIdx === i ? '#fff' : '#94a3b8',
                    }}
                  >
                    Phase {i+1}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '0.75rem',
          }}>
            {Object.entries(weights)
              .sort(([tA, wA], [tB, wB]) => {
                const isConsA = hardConstraints.has(tA);
                const isConsB = hardConstraints.has(tB);
                if (isConsA && !isConsB) return -1;
                if (!isConsA && isConsB) return 1;
                return wB - wA;
              })
              .map(([ticker, weight]) => (
                <div
                  key={ticker}
                  style={{
                    padding: '1rem 1.25rem', borderRadius: '0.75rem',
                    background: 'rgba(15, 23, 42, 0.6)', cursor: 'pointer',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    transition: 'border-color 0.2s, transform 0.2s',
                  }}
                  onClick={() => setEtfModal(ticker)}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(139,92,246,0.3)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {ticker}
                      {hardConstraints.has(ticker) && (
                        <span style={{
                          fontSize: '0.6rem', padding: '0.15rem 0.4rem', borderRadius: '0.3rem',
                          background: 'rgba(59,130,246,0.15)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.3)',
                        }}>
                          USER PREFERENCE
                        </span>
                      )}
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
        </div>
      ) : (
        <p style={{ color: '#64748b' }}>Select a portfolio from the sidebar to view details.</p>
      )}
      
      {etfModal && <AssetModal ticker={etfModal} onClose={() => setEtfModal(null)} />}
    </div>
  );
}
