import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useStore from '../../store';
import api from '../../api/client';

const FOMO_QUESTIONS = [
  {
    question: "A stock you've been watching drops 20% in a single day. What do you do?",
    options: [
      { label: 'Buy aggressively — it\'s on sale!', score: 3 },
      { label: 'Buy a small amount and watch', score: 2 },
      { label: 'Wait for it to stabilize', score: 1 },
    ],
  },
  {
    question: "Your coworker made 40% on a meme stock. Your reaction?",
    options: [
      { label: 'Research it and jump in immediately', score: 3 },
      { label: 'Take note but stick to my plan', score: 2 },
      { label: 'Ignore it — too risky', score: 1 },
    ],
  },
  {
    question: "A heavily hyped IPO launches tomorrow. You would:",
    options: [
      { label: 'Buy on day one', score: 3 },
      { label: 'Watch for a week then decide', score: 2 },
      { label: 'Never buy IPOs', score: 1 },
    ],
  },
  {
    question: "The market is at an all-time high. Your move?",
    options: [
      { label: 'Keep buying — momentum is everything', score: 3 },
      { label: 'Hold current positions', score: 2 },
      { label: 'Take some profits off the table', score: 1 },
    ],
  },
  {
    question: "A trending TikTok says to buy a particular crypto. You:",
    options: [
      { label: 'Throw in some money for fun', score: 3 },
      { label: 'Research it first, maybe small position', score: 2 },
      { label: 'Social media is not investment advice', score: 1 },
    ],
  },
];

const INITIAL_STATE = {
  goals: [{ name: '', amount: '', years: '' }],
  risk_tolerance: 50,
  fomo_answers: Array(FOMO_QUESTIONS.length).fill(null),
  hard_constraints: [{ ticker: '', pct: '' }],
  current_portfolio: [{ ticker: '', pct: '' }],
};

export default function QuestionnaireStepper() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(INITIAL_STATE);
  const [saving, setSaving] = useState(false);
  const navigate = useNavigate();
  const { fetchQuestionnaire, fetchUserData } = useStore();

  // Pre-fill from existing answers
  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/questionnaire/current');
        if (res.data.exists) {
          const a = res.data.answers;
          setForm((prev) => ({
            ...prev,
            goals: a.goals?.length ? a.goals : prev.goals,
            risk_tolerance: a.risk_tolerance ?? prev.risk_tolerance,
            hard_constraints: a.hard_constraints?.length
              ? a.hard_constraints.map((c) => ({ ticker: c.ticker, pct: c.pct }))
              : prev.hard_constraints,
            current_portfolio: a.current_portfolio?.length
              ? a.current_portfolio.map((c) => ({ ticker: c.ticker, pct: c.pct }))
              : prev.current_portfolio,
          }));
        }
      } catch (e) { /* first time user */ }
    })();
  }, []);

  const totalSteps = 5;

  // ─── Helpers ────────────────────────────────────────────────────────────
  const updateGoal = (idx, field, value) => {
    const next = [...form.goals];
    next[idx] = { ...next[idx], [field]: value };
    setForm({ ...form, goals: next });
  };
  const addGoal = () => setForm({ ...form, goals: [...form.goals, { name: '', amount: '', years: '' }] });
  const removeGoal = (idx) => setForm({ ...form, goals: form.goals.filter((_, i) => i !== idx) });

  const updateConstraint = (idx, field, value) => {
    const next = [...form.hard_constraints];
    next[idx] = { ...next[idx], [field]: value };
    setForm({ ...form, hard_constraints: next });
  };
  const addConstraint = () => setForm({ ...form, hard_constraints: [...form.hard_constraints, { ticker: '', pct: '' }] });

  const updatePortfolio = (idx, field, value) => {
    const next = [...form.current_portfolio];
    next[idx] = { ...next[idx], [field]: value };
    setForm({ ...form, current_portfolio: next });
  };
  const addPortfolioRow = () => setForm({ ...form, current_portfolio: [...form.current_portfolio, { ticker: '', pct: '' }] });

  const setFomoAnswer = (qIdx, score) => {
    const next = [...form.fomo_answers];
    next[qIdx] = score;
    setForm({ ...form, fomo_answers: next });
  };

  const fomoScore = Math.min(
    10,
    Math.max(1, Math.round(
      (form.fomo_answers.filter(Boolean).reduce((a, b) => a + b, 0) /
        Math.max(1, form.fomo_answers.filter(Boolean).length)) * 10 / 3
    ))
  );

  // ─── Submit ─────────────────────────────────────────────────────────────
  const handleSave = async () => {
    setSaving(true);
    try {
      const payload = {
        schema_version: 'v1',
        answers: {
          goals: form.goals.filter((g) => g.name && g.amount).map((g) => ({
            name: g.name,
            amount: parseFloat(g.amount),
            years: parseInt(g.years) || 5,
          })),
          risk_tolerance: form.risk_tolerance,
          fomo_tendency: fomoScore,
          hard_constraints: form.hard_constraints
            .filter((c) => c.ticker && c.pct)
            .map((c) => ({ ticker: c.ticker.toUpperCase(), pct: parseFloat(c.pct) })),
          current_portfolio: form.current_portfolio
            .filter((c) => c.ticker && c.pct)
            .map((c) => ({ ticker: c.ticker.toUpperCase(), pct: parseFloat(c.pct) })),
        },
      };
      await api.post('/questionnaire/save', payload);
      // Auto-generate a recommended portfolio from the saved answers
      await api.post('/recommend');
      await fetchQuestionnaire();
      await fetchUserData();
      navigate('/');
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  // ─── Shared Styles ─────────────────────────────────────────────────────
  const cardStyle = {
    maxWidth: 640,
    margin: '3rem auto',
    padding: '2rem',
    borderRadius: '1rem',
    background: 'rgba(15, 23, 42, 0.7)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255,255,255,0.08)',
    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
  };

  const inputStyle = {
    padding: '0.65rem 0.85rem',
    borderRadius: '0.4rem',
    background: 'rgba(30,41,59,0.8)',
    border: '1px solid rgba(255,255,255,0.1)',
    color: '#f1f5f9',
    fontSize: '0.9rem',
    outline: 'none',
    width: '100%',
    boxSizing: 'border-box',
  };

  const labelStyle = {
    color: '#94a3b8',
    fontSize: '0.78rem',
    fontWeight: 500,
    display: 'block',
    marginBottom: '0.35rem',
  };

  const btnPrimary = {
    padding: '0.7rem 1.5rem',
    borderRadius: '0.5rem',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    color: '#fff',
    fontWeight: 600,
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.9rem',
  };

  const btnSecondary = {
    padding: '0.7rem 1.5rem',
    borderRadius: '0.5rem',
    background: 'rgba(255,255,255,0.05)',
    color: '#94a3b8',
    fontWeight: 500,
    border: '1px solid rgba(255,255,255,0.1)',
    cursor: 'pointer',
    fontSize: '0.9rem',
  };

  // ─── Steps ─────────────────────────────────────────────────────────────
  const renderStep = () => {
    switch (step) {
      case 0: // Financial Goals
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Financial Goals
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              Define your investment objectives. Short-term goals (≤5 years) will be routed to safer investments.
            </p>
            {form.goals.map((g, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '2fr 1fr 1fr auto', gap: '0.5rem',
                marginBottom: '0.75rem', alignItems: 'end',
              }}>
                <div>
                  <label style={labelStyle}>Goal Name</label>
                  <input style={inputStyle} value={g.name} placeholder="e.g. House Downpayment"
                    onChange={(e) => updateGoal(i, 'name', e.target.value)} />
                </div>
                <div>
                  <label style={labelStyle}>Amount ($)</label>
                  <input style={inputStyle} type="number" value={g.amount} placeholder="200000"
                    onChange={(e) => updateGoal(i, 'amount', e.target.value)} />
                </div>
                <div>
                  <label style={labelStyle}>Years</label>
                  <input style={inputStyle} type="number" value={g.years} placeholder="3"
                    onChange={(e) => updateGoal(i, 'years', e.target.value)} />
                </div>
                {form.goals.length > 1 && (
                  <button onClick={() => removeGoal(i)} style={{
                    background: 'none', border: 'none', color: '#f87171', cursor: 'pointer',
                    fontSize: '1.2rem', paddingBottom: '0.5rem',
                  }}>×</button>
                )}
              </div>
            ))}
            <button onClick={addGoal} style={{
              ...btnSecondary, marginTop: '0.5rem', fontSize: '0.8rem',
            }}>+ Add Another Goal</button>
          </div>
        );

      case 1: // Risk Tolerance
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Risk Tolerance
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '2rem' }}>
              How much market volatility can you handle? 1 = extremely conservative, 100 = very aggressive.
            </p>
            <div style={{ textAlign: 'center' }}>
              <span style={{
                fontSize: '3rem', fontWeight: 700,
                background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
                {form.risk_tolerance}
              </span>
              <input
                type="range"
                min="1" max="100"
                value={form.risk_tolerance}
                onChange={(e) => setForm({ ...form, risk_tolerance: parseInt(e.target.value) })}
                style={{ width: '100%', marginTop: '1rem', accentColor: '#3b82f6' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#64748b', fontSize: '0.75rem' }}>
                <span>Conservative</span>
                <span>Moderate</span>
                <span>Aggressive</span>
              </div>
            </div>
          </div>
        );

      case 2: // FOMO Questions
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Investment Behavior
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              Answer these situational questions so we can gauge your FOMO tendency.
              <span style={{ color: '#60a5fa', fontWeight: 600 }}> Current FOMO Score: {fomoScore}/10</span>
            </p>
            {FOMO_QUESTIONS.map((q, qi) => (
              <div key={qi} style={{
                marginBottom: '1.25rem',
                padding: '1rem',
                borderRadius: '0.6rem',
                background: 'rgba(30,41,59,0.5)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>
                <p style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 500, marginBottom: '0.75rem' }}>
                  {qi + 1}. {q.question}
                </p>
                {q.options.map((opt, oi) => (
                  <label key={oi} style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    padding: '0.4rem 0.6rem', borderRadius: '0.4rem',
                    cursor: 'pointer', marginBottom: '0.3rem',
                    background: form.fomo_answers[qi] === opt.score
                      ? 'rgba(59,130,246,0.12)' : 'transparent',
                    border: form.fomo_answers[qi] === opt.score
                      ? '1px solid rgba(59,130,246,0.3)' : '1px solid transparent',
                    transition: 'all 0.15s',
                  }}>
                    <input
                      type="radio"
                      name={`fomo-${qi}`}
                      checked={form.fomo_answers[qi] === opt.score}
                      onChange={() => setFomoAnswer(qi, opt.score)}
                      style={{ accentColor: '#3b82f6' }}
                    />
                    <span style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>{opt.label}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        );

      case 3: // Hard Constraints
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Specific Stock Preferences
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              Want to guarantee a position in specific companies? These will be carved out before optimization.
            </p>
            {form.hard_constraints.map((c, i) => (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <div>
                  <label style={labelStyle}>Ticker</label>
                  <input style={inputStyle} value={c.ticker} placeholder="AAPL"
                    onChange={(e) => updateConstraint(i, 'ticker', e.target.value)} />
                </div>
                <div>
                  <label style={labelStyle}>Allocation %</label>
                  <input style={inputStyle} type="number" value={c.pct} placeholder="10"
                    onChange={(e) => updateConstraint(i, 'pct', e.target.value)} />
                </div>
              </div>
            ))}
            <button onClick={addConstraint} style={{ ...btnSecondary, fontSize: '0.8rem', marginTop: '0.5rem' }}>
              + Add Ticker
            </button>
          </div>
        );

      case 4: // Current Portfolio
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Current Portfolio (Optional)
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              If you already hold investments, enter them here so we can compare and optimize.
            </p>
            {form.current_portfolio.map((c, i) => (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <div>
                  <label style={labelStyle}>Ticker</label>
                  <input style={inputStyle} value={c.ticker} placeholder="VOO"
                    onChange={(e) => updatePortfolio(i, 'ticker', e.target.value)} />
                </div>
                <div>
                  <label style={labelStyle}>Allocation %</label>
                  <input style={inputStyle} type="number" value={c.pct} placeholder="50"
                    onChange={(e) => updatePortfolio(i, 'pct', e.target.value)} />
                </div>
              </div>
            ))}
            <button onClick={addPortfolioRow} style={{ ...btnSecondary, fontSize: '0.8rem', marginTop: '0.5rem' }}>
              + Add Holding
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%)',
      padding: '1rem',
    }}>
      {/* Progress Bar */}
      <div style={{ maxWidth: 640, margin: '0 auto 0.5rem' }}>
        <div style={{
          display: 'flex', gap: '0.25rem',
        }}>
          {Array.from({ length: totalSteps }).map((_, i) => (
            <div key={i} style={{
              flex: 1, height: 4, borderRadius: 2,
              background: i <= step
                ? 'linear-gradient(90deg, #3b82f6, #8b5cf6)'
                : 'rgba(255,255,255,0.08)',
              transition: 'background 0.3s',
            }} />
          ))}
        </div>
        <p style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '0.5rem' }}>
          Step {step + 1} of {totalSteps}
        </p>
      </div>

      <div style={cardStyle}>
        {renderStep()}

        {/* Navigation Buttons */}
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          marginTop: '2rem',
        }}>
          <button
            onClick={() => step > 0 ? setStep(step - 1) : navigate('/')}
            style={btnSecondary}
          >
            {step === 0 ? 'Cancel' : 'Back'}
          </button>

          {step < totalSteps - 1 ? (
            <button onClick={() => setStep(step + 1)} style={btnPrimary}>
              Next
            </button>
          ) : (
            <button onClick={handleSave} disabled={saving} style={{
              ...btnPrimary, opacity: saving ? 0.7 : 1,
            }}>
              {saving ? 'Saving...' : 'Complete & Generate'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
