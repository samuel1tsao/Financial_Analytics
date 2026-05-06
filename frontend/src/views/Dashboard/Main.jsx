import { useState, useEffect, useMemo } from 'react';
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
        Current: {fmt(d.expected)}
      </p>

      {payload.filter(p => p.dataKey.startsWith('comp_')).map((p, i) => (
        <p key={i} style={{ color: p.stroke, fontSize: '0.8rem', marginTop: '0.2rem', fontWeight: 500 }}>
          {p.name}: {fmt(p.value)}
        </p>
      ))}

      <div style={{ marginTop: '0.6rem', paddingTop: '0.4rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <p style={{ color: 'rgba(139,92,246,0.95)', fontSize: '0.75rem', marginBottom: '0.1rem' }}>40th-60th Range: {fmt(d.p40)} – {fmt(d.p60)}</p>
        <p style={{ color: 'rgba(139,92,246,0.85)', fontSize: '0.75rem', marginBottom: '0.1rem' }}>30th-70th Range: {fmt(d.p30)} – {fmt(d.p70)}</p>
        <p style={{ color: 'rgba(139,92,246,0.75)', fontSize: '0.75rem', marginBottom: '0.1rem' }}>20th-80th Range: {fmt(d.p20)} – {fmt(d.p80)}</p>
        <p style={{ color: 'rgba(139,92,246,0.65)', fontSize: '0.75rem' }}>10th-90th Range: {fmt(d.p10)} – {fmt(d.p90)}</p>
      </div>
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
  const [sandboxStartCap, setSandboxStartCap] = useState(100000);
  const [sandboxMonthlyContrib, setSandboxMonthlyContrib] = useState(500);
  
  // ETF Modal
  const [etfModal, setEtfModal] = useState(null);
  const [activeSegmentIdx, setActiveSegmentIdx] = useState(0);
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);

  // Comparison State
  const [selectedComparisons, setSelectedComparisons] = useState([]); // Array of benchmark keys or portfolio IDs
  const [comparisonSims, setComparisonSims] = useState({});

  const activePortfolio = useMemo(() => {
    // If we have a specific ID selected, strictly use it. 
    // Do NOT fallback to is_current if the requested ID isn't in the list yet (wait for sync)
    if (activePortfolioId) {
      return portfolios.find((p) => p.id === activePortfolioId) || null;
    }
    // Otherwise, fallback to the one marked as current
    return portfolios.find((p) => p.is_current) || null;
  }, [portfolios, activePortfolioId]);

  useEffect(() => {
    if (activePortfolio && activePortfolio.weights) {
      const data = activePortfolio.weights;
      if (data.goals) {
        setCustomGoals(data.goals);
        setDebouncedGoals(data.goals);
      }
      if (data.start_cap) setSandboxStartCap(data.start_cap);
      else if (questionnaire?.start_cap) setSandboxStartCap(questionnaire.start_cap);
      
      if (data.monthly_contrib) setSandboxMonthlyContrib(data.monthly_contrib);
      else if (questionnaire?.monthly_contrib) setSandboxMonthlyContrib(questionnaire.monthly_contrib);
    }
  }, [activePortfolio?.id]);

  // ─── Local Recalculation Logic ─────────────────────────────────────────────
  const calculateLocalProjection = (baseSim, goals, startCapital, monthlyInc, projectionYears = 30) => {
    if (!baseSim || !baseSim.expected_annual_returns) return baseSim;

    const totalYears = projectionYears;
    const goalMap = {};
    goals.forEach(g => {
      goalMap[g.years] = (goalMap[g.years] || 0) + g.amount;
    });

    const stepBalances = [];
    const cashOutEvents = [];

    const runPath = (annualReturns, isExpected) => {
      let balance = Number(startCapital) || 0;
      const path = [balance];
      const contribAnnual = (Number(monthlyInc) || 0) * 12;

      for (let yr = 1; yr <= totalYears; yr++) {
        // Fallback to last known return if we extend past the simulation data
        // This prevents NaN while waiting for the background re-fetch
        const r = (yr - 1 < annualReturns.length) 
          ? annualReturns[yr - 1] 
          : (annualReturns.length > 0 ? annualReturns[annualReturns.length - 1] : 0.07);
        
        balance *= (1 + r);
        balance += contribAnnual;

        const needed = goalMap[yr] || 0;
        if (needed > 0) {
          let withdrawn = 0;
          let shortfall = 0;

          if (balance >= needed) {
            withdrawn = needed;
            balance -= needed;
          } else {
            withdrawn = balance;
            shortfall = needed - balance;
            balance = 0;
          }

          if (isExpected) {
            stepBalances.push({ 
              year: yr, 
              balance: balance,
              shortfall: shortfall,
              withdrawn: withdrawn,
              goal_amount: needed
            });
            const goalForYear = goals.find(g => g.years === yr) || { name: 'Goal' };
            cashOutEvents.push({
              year: yr,
              amount: withdrawn,
              shortfall: shortfall,
              goal_name: goalForYear.name,
              remaining_expected: balance
            });
          }
        }
        path.push(balance);
      }
      return path;
    };

    const expectedPath = runPath(baseSim.expected_annual_returns || [], true);
    
    // True Percentile Calculation
    const percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    const percentilePaths = {};
    percentiles.forEach(p => percentilePaths[`p${p}`] = []);

    if (baseSim.raw_paths && baseSim.raw_paths.length > 0) {
      const allWealthPaths = baseSim.raw_paths.map(pathReturns => runPath(pathReturns, false));
      
      for (let yr = 0; yr <= totalYears; yr++) {
        const balancesForYear = allWealthPaths.map(p => p[yr]).sort((a, b) => a - b);
        percentiles.forEach(p => {
          const idx = Math.floor((p / 100) * balancesForYear.length);
          percentilePaths[`p${p}`].push(balancesForYear[Math.min(idx, balancesForYear.length - 1)]);
        });
      }
    } else {
      // Fallback
      percentiles.forEach(p => percentilePaths[`p${p}`] = expectedPath);
    }

    const goalAnnotations = goals.map(g => ({
      year: g.years,
      label: g.name,
      amount: g.amount,
      is_short_term: g.years <= 5,
    }));

    // Update stats (rough approximation for CAGR)
    const finalExp = expectedPath.length > 0 ? expectedPath[expectedPath.length - 1] : 0;
    const totalWithdrawn = goals.reduce((sum, g) => sum + (Number(g.amount) || 0), 0);
    const startCapNum = Number(startCapital) || 1;
    const cagrBase = (startCapNum > 0 && totalYears > 0) ? (finalExp + totalWithdrawn) / startCapNum : 0;
    const cagr = (cagrBase > 0) ? (Math.pow(cagrBase, 1 / totalYears) - 1) * 100 : 0;

    return {
      ...baseSim,
      years: Array.from({ length: totalYears + 1 }, (_, i) => i),
      expected_path: percentilePaths.p50, // Use Median as Expected
      ...percentilePaths,
      step_balances: stepBalances,
      cash_out_events: cashOutEvents.sort((a,b) => a.year - b.year),
      goal_annotations: goalAnnotations,
      portfolio_stats: {
        ...(baseSim.portfolio_stats || {}),
        projected_final_expected: percentilePaths.p50[percentilePaths.p50.length - 1],
        projected_final_upper: percentilePaths.p90[percentilePaths.p90.length - 1],
        projected_final_lower: percentilePaths.p10[percentilePaths.p10.length - 1],
        expected_annual_return: (isNaN(cagr) || !isFinite(cagr)) ? "0.00" : cagr.toFixed(2),
      },
      segment_stats: baseSim.segment_stats,
    };
  };

  let weights = {};
  let segments = [];
  let rationales = {};
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
      rationales = currentSeg?.rationales || {};
    } else if (data && data.segments) {
      segments = data.segments;
      const currentSeg = segments[activeSegmentIdx] || segments[0];
      weights = currentSeg?.weights || {};
      rationales = currentSeg?.rationales || {};
    } else {
      weights = data || {};
      rationales = {};
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
    
    const maxGoalYear = debouncedGoals.length > 0 ? Math.max(30, ...debouncedGoals.map(g => Number(g.years))) : 30;
    const currentHorizon = simulation?.expected_annual_returns?.length || 0;
    
    const isStructuralChange = simulation === null || 
                               simulation.error ||
                               !simulation.expected_annual_returns ||
                               JSON.stringify(debouncedGoals.map(g => g.years)) !== JSON.stringify(simulation.goal_annotations?.map(a => a.year)) ||
                               currentHorizon < maxGoalYear;
    
    if (!isStructuralChange && simulation) {
      return;
    }

    let cancelled = false;
    (async () => {
      const cacheKey = `${activePortfolio.id}_${JSON.stringify(debouncedGoals)}`;
      
      if (!activePortfolio.id) {
        console.warn("Active portfolio has no ID, skipping simulation fetch");
        return;
      }
      
        const maxGoalYear = Math.max(30, ...debouncedGoals.map(g => Number(g.years)));

        setSimLoading(true);
        setSimError('');
        try {
          const res = await api.post('/simulate', {
            portfolio_id: activePortfolio.id,
            initial_investment: startCap,
            monthly_contrib: monthlyContrib,
            projection_years: maxGoalYear,
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
  }, [activePortfolio?.id, debouncedGoals, sandboxStartCap, sandboxMonthlyContrib]);

  // ─── Comparison Fetching ──────────────────────────────────────────────────
  useEffect(() => {
    const maxGoalYear = Math.max(30, ...customGoals.map(g => Number(g.years)));
    selectedComparisons.forEach(async (id) => {
      // Re-fetch if cached version is too short
      if (comparisonSims[id] && comparisonSims[id].expected_annual_returns?.length >= maxGoalYear) return;
      
      try {
        let res;
        if (id === 'sp500' || id === 'conservative_60_40') {
          res = await api.post('/simulate_benchmark', {
            benchmark_type: id,
            projection_years: maxGoalYear
          });
          setComparisonSims(prev => ({ ...prev, [id]: res.data }));
        } else {
          res = await api.post('/simulate', {
            portfolio_id: id,
            initial_investment: sandboxStartCap,
            monthly_contrib: sandboxMonthlyContrib,
            projection_years: maxGoalYear,
          });
          setComparisonSims(prev => ({ ...prev, [id]: res.data.simulation }));
        }
      } catch (err) {
        console.error("Failed to fetch comparison:", id, err);
      }
    });
  }, [selectedComparisons, sandboxStartCap, sandboxMonthlyContrib, customGoals]);

  // Apply local recalculation to comparisons
  const activeComparisons = useMemo(() => {
    const maxGoalYear = Math.max(30, ...customGoals.map(g => Number(g.years)));
    return selectedComparisons.map(id => {
      const baseSim = comparisonSims[id];
      if (!baseSim) return null;
      return {
        id,
        name: id === 'sp500' ? 'S&P 500' : id === 'conservative_60_40' ? '60/40 Benchmark' : (portfolios.find(p => p.id === id)?.profile_name || 'Portfolio'),
        data: calculateLocalProjection(baseSim, customGoals, sandboxStartCap, sandboxMonthlyContrib, maxGoalYear)
      };
    }).filter(Boolean);
  }, [selectedComparisons, comparisonSims, customGoals, sandboxStartCap, sandboxMonthlyContrib]);

  // Apply local recalculation to the simulation data
  const activeSimulation = useMemo(() => {
    const maxGoalYear = Math.max(30, ...customGoals.map(g => Number(g.years)));
    return calculateLocalProjection(simulation, customGoals, sandboxStartCap, sandboxMonthlyContrib, maxGoalYear);
  }, [simulation, customGoals, sandboxStartCap, sandboxMonthlyContrib]);

  // Sync back local simulation to stats
  const chartData = useMemo(() => {
    if (!activeSimulation || activeSimulation.error) return [];
    
    return activeSimulation.years.map((yr, i) => {
      const row = {
        year: yr,
        expected: activeSimulation.expected_path[i],
        p10: activeSimulation.p10[i],
        p20: activeSimulation.p20[i],
        p30: activeSimulation.p30[i],
        p40: activeSimulation.p40[i],
        p50: activeSimulation.p50[i],
        p60: activeSimulation.p60[i],
        p70: activeSimulation.p70[i],
        p80: activeSimulation.p80[i],
        p90: activeSimulation.p90[i],
        band_80_90: [activeSimulation.p80[i], activeSimulation.p90[i]],
        band_70_80: [activeSimulation.p70[i], activeSimulation.p80[i]],
        band_60_70: [activeSimulation.p60[i], activeSimulation.p70[i]],
        band_50_60: [activeSimulation.p50[i], activeSimulation.p60[i]],
        band_40_50: [activeSimulation.p40[i], activeSimulation.p50[i]],
        band_30_40: [activeSimulation.p30[i], activeSimulation.p40[i]],
        band_20_30: [activeSimulation.p20[i], activeSimulation.p30[i]],
        band_10_20: [activeSimulation.p10[i], activeSimulation.p20[i]],
      };
      
      activeComparisons.forEach(comp => {
        row[`comp_${comp.id}`] = comp.data.expected_path[i];
      });
      
      return row;
    });
  }, [activeSimulation, activeComparisons]);

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

      {/* Stats Summary Row (Always visible shell) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '0.75rem', marginBottom: '1.5rem',
      }}>
        <StatCard
          label={`Expected Return ${segments.length > 1 ? `(Phase ${activeSegmentIdx + 1})` : ''}`}
          value={simLoading && !activeSimulation ? "..." : `${activeSimulation?.segment_stats?.[activeSegmentIdx]?.expected_return || stats.expected_annual_return}%`}
          subtext={segments.length > 1 ? "Selected phase growth" : "Annual average"}
          color="#4ade80"
        />
        <StatCard
          label={`Volatility ${segments.length > 1 ? `(Phase ${activeSegmentIdx + 1})` : ''}`}
          value={simLoading && !activeSimulation ? "..." : `${activeSimulation?.segment_stats?.[activeSegmentIdx]?.volatility || stats.annual_volatility}%`}
          subtext={segments.length > 1 ? "Selected phase risk" : "Annual std dev"}
          color="#facc15"
        />
        <StatCard
          label={`Sharpe Ratio ${segments.length > 1 ? `(Phase ${activeSegmentIdx + 1})` : ''}`}
          value={simLoading && !activeSimulation ? "..." : (activeSimulation?.segment_stats?.[activeSegmentIdx]?.sharpe_ratio ?? stats.sharpe_ratio)}
          subtext="Risk-adjusted return"
          color="#60a5fa"
        />
        <StatCard
          label={`Projected (${activeSimulation?.years?.length ? activeSimulation.years.length - 1 : 30}yr)`}
          value={simLoading && !activeSimulation ? "..." : fmt(stats.projected_final_expected || 0)}
          subtext={simLoading && !activeSimulation ? "Calculating..." : `Range: ${fmt(stats.projected_final_lower || 0)} — ${fmt(stats.projected_final_upper || 0)}`}
          color="#c084fc"
        />
      </div>

      {/* View Toggle (Always visible) */}
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
          {activeSimulation?.years?.length ? `${activeSimulation.years.length - 1}Y` : '30Y'} PROJECTION
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
          display: 'flex', flexDirection: 'column', gap: '1.5rem'
        }}>
          <div style={{
            padding: '1.5rem', borderRadius: '0.75rem',
            background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.06)',
            position: 'relative'
          }}>
          {simLoading && (
            <div style={{
              position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
              background: 'rgba(15,23,42,0.4)', backdropFilter: 'blur(4px)',
              zIndex: 10, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', color: '#64748b',
              borderRadius: '0.75rem',
            }}>
              <div style={{
                width: 30, height: 30, marginBottom: '0.75rem',
                borderRadius: '50%', border: '3px solid rgba(59,130,246,0.2)',
                borderTopColor: '#3b82f6',
                animation: 'spin 1s linear infinite',
              }} />
              <p style={{ fontSize: '0.8rem' }}>Updating simulation...</p>
            </div>
          )}

          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: '1rem',
          }}>
            <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem', margin: 0 }}>
              {viewMode === 'projection' ? 'Portfolio Growth Projection' : '10-Year Historical Backtest'}
            </h3>
            <span style={{ color: '#64748b', fontSize: '0.75rem' }}>
              {viewMode === 'projection' ? `±2σ Confidence · ${activeSimulation?.years?.length ? activeSimulation.years.length - 1 : 30} Year Horizon` : `Actual Returns · ${backtestData?.stats?.start_date} to Present`}
            </span>
          </div>

          {/* Comparison Selector */}
          {viewMode === 'projection' && (
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
              <span style={{ color: '#64748b', fontSize: '0.7rem', alignSelf: 'center', marginRight: '0.5rem' }}>COMPARE:</span>
              {[
                { id: 'sp500', label: 'S&P 500', color: '#f59e0b' },
                { id: 'conservative_60_40', label: '60/40 Bench', color: '#10b981' },
                ...portfolios.filter(p => p.id !== activePortfolio?.id).map(p => ({ id: p.id, label: p.profile_name, color: '#ec4899' }))
              ].map(opt => {
                const isSelected = selectedComparisons.includes(opt.id);
                return (
                  <button
                    key={opt.id}
                    onClick={() => {
                      if (isSelected) setSelectedComparisons(prev => prev.filter(x => x !== opt.id));
                      else setSelectedComparisons(prev => [...prev, opt.id]);
                    }}
                    style={{
                      background: isSelected ? `${opt.color}22` : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${isSelected ? opt.color : 'rgba(255,255,255,0.1)'}`,
                      color: isSelected ? opt.color : '#64748b',
                      padding: '0.25rem 0.75rem', borderRadius: '2rem', fontSize: '0.65rem',
                      fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s'
                    }}
                  >
                    {isSelected ? '✓ ' : '+ '}{opt.label}
                  </button>
                );
              })}
            </div>
          )}

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
                    
                    {/* Upper Bands */}
                    <Area type="monotone" dataKey="band_80_90" stroke="none" fill="rgba(139,92,246,0.05)" />
                    <Area type="monotone" dataKey="band_70_80" stroke="none" fill="rgba(139,92,246,0.15)" />
                    <Area type="monotone" dataKey="band_60_70" stroke="none" fill="rgba(139,92,246,0.30)" />
                    <Area type="monotone" dataKey="band_50_60" stroke="none" fill="rgba(139,92,246,0.50)" />

                    {/* Lower Bands */}
                    <Area type="monotone" dataKey="band_40_50" stroke="none" fill="rgba(139,92,246,0.50)" />
                    <Area type="monotone" dataKey="band_30_40" stroke="none" fill="rgba(139,92,246,0.30)" />
                    <Area type="monotone" dataKey="band_20_30" stroke="none" fill="rgba(139,92,246,0.15)" />
                    <Area type="monotone" dataKey="band_10_20" stroke="none" fill="rgba(139,92,246,0.05)" />
                    
                    <Area type="monotone" dataKey="expected" name="Median Path" stroke="#3b82f6" strokeWidth={3} fill="url(#gradExpected)" dot={false} />
                    
                    {activeComparisons.map((comp, idx) => {
                      const colors = ['#f59e0b', '#10b981', '#ec4899', '#8b5cf6', '#06b6d4'];
                      const color = colors[idx % colors.length];
                      return (
                        <Line 
                          key={comp.id} 
                          type="monotone" 
                          dataKey={`comp_${comp.id}`} 
                          name={comp.name} 
                          stroke={color} 
                          strokeWidth={2} 
                          dot={false} 
                          strokeDasharray="5 5"
                        />
                      );
                    })}

                    {activeSimulation?.goal_annotations?.map((g, i) => (
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

          <BalanceStepTable 
            steps={activeSimulation?.step_balances} 
            segments={segments}
            portfolioName={activePortfolio?.profile_name || 'Portfolio'} 
          />
        </div>
        
        <div style={{
          flex: 1, minWidth: 280,
          display: 'flex', flexDirection: 'column', gap: '1rem'
        }}>
          <div style={{
            padding: '1.5rem', borderRadius: '0.75rem',
            background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
              <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.9rem', marginBottom: '1rem' }}>
                Interactive Sandbox
              </h3>
              
              {/* Global Portfolio Parameters */}
              <div style={{ marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 600, display: 'block', marginBottom: '0.4rem' }}>
                    INITIAL INVESTMENT ($)
                  </label>
                  <input 
                    type="number"
                    value={sandboxStartCap}
                    onChange={e => setSandboxStartCap(Number(e.target.value))}
                    style={{ 
                      width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                      color: '#f1f5f9', padding: '0.5rem', borderRadius: '0.4rem', fontSize: '0.85rem',
                      outline: 'none'
                    }}
                    onFocus={e => e.target.style.borderColor = '#3b82f6'}
                    onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
                  />
                </div>
                
                <div style={{ marginBottom: '0.5rem' }}>
                  <label style={{ color: '#94a3b8', fontSize: '0.7rem', fontWeight: 600, display: 'block', marginBottom: '0.4rem' }}>
                    MONTHLY DEPOSIT ($)
                  </label>
                  <input 
                    type="number"
                    value={sandboxMonthlyContrib}
                    onChange={e => setSandboxMonthlyContrib(Number(e.target.value))}
                    style={{ 
                      width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                      color: '#f1f5f9', padding: '0.5rem', borderRadius: '0.4rem', fontSize: '0.85rem',
                      outline: 'none'
                    }}
                    onFocus={e => e.target.style.borderColor = '#3b82f6'}
                    onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
                  />
                </div>
              </div>

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
                    <label style={{ color: '#64748b', fontSize: '0.7rem' }}>Amount ($)</label>
                  </div>
                  <input 
                    type="number"
                    value={g.amount}
                    onChange={e => {
                      const ng = [...customGoals];
                      ng[i].amount = Number(e.target.value);
                      setCustomGoals(ng);
                    }}
                    style={{ 
                      width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                      color: '#f1f5f9', padding: '0.4rem', borderRadius: '0.3rem', fontSize: '0.85rem', marginBottom: '0.8rem',
                      outline: 'none'
                    }}
                  />

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                    <label style={{ color: '#64748b', fontSize: '0.7rem' }}>Target Year</label>
                  </div>
                  <input 
                    type="number"
                    value={g.years}
                    onChange={e => {
                      const ng = [...customGoals];
                      ng[i].years = Number(e.target.value);
                      setCustomGoals(ng);
                    }}
                    style={{ 
                      width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                      color: '#f1f5f9', padding: '0.4rem', borderRadius: '0.3rem', fontSize: '0.85rem', marginBottom: '0.8rem',
                      outline: 'none'
                    }}
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

              {activeSimulation?.cash_out_events?.length > 0 && (
                <div style={{ marginTop: '1rem', background: 'rgba(245,158,11,0.08)', padding: '0.75rem', borderRadius: '0.4rem', border: '1px solid rgba(245,158,11,0.15)' }}>
                  <p style={{ color: '#f59e0b', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                    CASH-OUT EVENT LOG
                  </p>
                  {activeSimulation.cash_out_events.map((ev, i) => (
                    <div key={i} style={{ marginBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '0.3rem' }}>
                      <p style={{ color: '#e2e8f0', fontSize: '0.75rem', margin: '0.2rem 0' }}>
                        <strong>Yr {ev.year}</strong>: {ev.goal_name} 
                        <span style={{ color: ev.shortfall > 0 ? '#fca5a5' : '#4ade80', marginLeft: '0.4rem' }}>
                          (-{fmt(ev.amount)})
                        </span>
                      </p>
                      {ev.shortfall > 0 && (
                        <p style={{ color: '#ef4444', fontSize: '0.65rem', margin: 0, fontWeight: 600 }}>
                          ⚠ SHORTFALL: {fmt(ev.shortfall)}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
      </div>

      {/* Portfolio Allocation Cards */}
      {activePortfolio ? (
        <div style={{ marginTop: '2rem' }}>
          <>
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
                    {rationales[ticker] && (
                      <div style={{ marginTop: '0.75rem', fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.4, fontStyle: 'italic' }}>
                        "{rationales[ticker]}"
                      </div>
                    )}
                  </div>
                ))}
            </div>
          </>
        </div>
      ) : (
        <p style={{ color: '#64748b' }}>Select a portfolio from the sidebar to view details.</p>
      )}
      
      {etfModal && <AssetModal ticker={etfModal} onClose={() => setEtfModal(null)} />}
    </div>
  );
}
