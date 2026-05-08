import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useStore from '../../store';
import api from '../../api/client';
import TickerInput from '../../components/TickerInput';

const DRAWDOWN_QUESTIONS = [
  {
    question: "Your portfolio drops 30% during a market crash. How do you react?",
    options: [
      { label: "Buy more — this is a sale!", score: 2 },
      { label: "Hold steady, it'll recover", score: 5 },
      { label: "Sell some to limit losses", score: 7 },
      { label: "Sell everything immediately", score: 9 },
    ],
  },
  {
    question: "Markets drop 15% over 3 months. Your friend suggests going all-cash. You:",
    options: [
      { label: "Disagree — I'd increase my positions", score: 2 },
      { label: "Disagree — I'd stay the course", score: 4 },
      { label: "Consider it — I'd sell my riskiest holdings", score: 7 },
      { label: "Agree — preserving capital is priority #1", score: 9 },
    ],
  },
  {
    question: "In 2008, the S&P 500 fell ~50%. If that happened today, you would:",
    options: [
      { label: "See it as a once-in-a-decade opportunity to buy", score: 2 },
      { label: "Be uncomfortable but not change anything", score: 5 },
      { label: "Reduce equity exposure significantly", score: 8 },
      { label: "Exit the market entirely until recovery", score: 10 },
    ],
  },
];

const VOLATILITY_QUESTIONS = [
  {
    question: "Which portfolio do you prefer over 5 years?",
    options: [
      { label: "Portfolio A: +7% every year (consistent, lower total)", score: 8 },
      { label: "Portfolio B: -10%, +35%, +5%, +25%, -5% (~10% avg, volatile)", score: 3 },
      { label: "No preference", score: 5 },
    ],
  },
  {
    question: "Two funds have the same 10-year average return of 10%. Fund X had steady growth. Fund Y had wild swings. You pick:",
    options: [
      { label: "Fund X — I want the smooth ride", score: 8 },
      { label: "Either is fine if the average is the same", score: 5 },
      { label: "Fund Y — those swings mean bigger upside potential", score: 2 },
    ],
  },
  {
    question: "Your portfolio is up 25% this year but was down 20% last year. How do you feel?",
    options: [
      { label: "Great — I'm ahead overall", score: 2 },
      { label: "Uneasy — I wish it were more predictable", score: 6 },
      { label: "Anxious — I need more consistency", score: 9 },
    ],
  },
];

const INITIAL_STATE = {
  goals: [{ name: '', amount: '', years: '' }],
  drawdown_answers: Array(DRAWDOWN_QUESTIONS.length).fill(null),
  volatility_answers: Array(VOLATILITY_QUESTIONS.length).fill(null),
  goal_flexibility: 5,
  concentration_pref: 5,
  start_cap: 100000,
  monthly_contrib: 500,
  hard_constraints: [{ ticker: '', pct: '' }],
  current_portfolio: [{ ticker: '', pct: '' }],
};

// validity maps: { "rowKey": true | false | null }
// null = not yet checked / empty
const INITIAL_VALIDITY = { hard_constraints: {}, current_portfolio: {} };

export default function QuestionnaireStepper() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(INITIAL_STATE);
  const [validity, setValidity] = useState(INITIAL_VALIDITY);
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
            drawdown_answers: a.drawdown_answers?.length ? a.drawdown_answers : prev.drawdown_answers,
            volatility_answers: a.volatility_answers?.length ? a.volatility_answers : prev.volatility_answers,
            goal_flexibility: a.goal_flexibility ?? prev.goal_flexibility,
            concentration_pref: a.concentration_pref ?? prev.concentration_pref,
            start_cap: a.start_cap ?? prev.start_cap,
            monthly_contrib: a.monthly_contrib ?? prev.monthly_contrib,
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

  const totalSteps = 7;

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
  const addConstraint = () => {
    setForm({ ...form, hard_constraints: [...form.hard_constraints, { ticker: '', pct: '' }] });
    // new row starts as null validity
    setValidity(v => ({ ...v, hard_constraints: { ...v.hard_constraints, [form.hard_constraints.length]: null } }));
  };

  const updatePortfolio = (idx, field, value) => {
    const next = [...form.current_portfolio];
    next[idx] = { ...next[idx], [field]: value };
    setForm({ ...form, current_portfolio: next });
  };
  const addPortfolioRow = () => {
    setForm({ ...form, current_portfolio: [...form.current_portfolio, { ticker: '', pct: '' }] });
    setValidity(v => ({ ...v, current_portfolio: { ...v.current_portfolio, [form.current_portfolio.length]: null } }));
  };

  const removeConstraint = (idx) => {
    const next = form.hard_constraints.filter((_, i) => i !== idx);
    setForm({ ...form, hard_constraints: next.length ? next : [{ ticker: '', pct: '' }] });
    // Also clean up validity
    const nextV = {};
    next.forEach((_, i) => { nextV[i] = validity.hard_constraints[i]; });
    setValidity(v => ({ ...v, hard_constraints: nextV }));
  };

  const removePortfolioRow = (idx) => {
    const next = form.current_portfolio.filter((_, i) => i !== idx);
    setForm({ ...form, current_portfolio: next.length ? next : [{ ticker: '', pct: '' }] });
    // Also clean up validity
    const nextV = {};
    next.forEach((_, i) => { nextV[i] = validity.current_portfolio[i]; });
    setValidity(v => ({ ...v, current_portfolio: nextV }));
  };

  // Validity helpers
  const setConstraintValid = (idx, ticker, isValid) => {
    setValidity(v => ({ ...v, hard_constraints: { ...v.hard_constraints, [idx]: isValid } }));
  };
  const setPortfolioValid = (idx, ticker, isValid) => {
    setValidity(v => ({ ...v, current_portfolio: { ...v.current_portfolio, [idx]: isValid } }));
  };

  // Helper for specific error messaging in JSX
  const constraintsOk = form.hard_constraints.every((c, i) =>
    (!c.ticker && !c.pct) || (c.ticker && parseFloat(c.pct) > 0 && validity.hard_constraints[i] !== false)
  );
  const portfolioOk = form.current_portfolio.every((c, i) =>
    (!c.ticker && !c.pct) || (c.ticker && parseFloat(c.pct) > 0 && validity.current_portfolio[i] !== false)
  );

  // Returns true if the current step is valid for moving forward
  const stepOk = (() => {
    switch (step) {
      case 0: // Goals
        return parseFloat(form.start_cap) > 0 && 
               form.goals.some(g => g.name.trim() !== '' && parseFloat(g.amount) > 0 && parseInt(g.years) > 0);
      case 1: // Drawdown
        return form.drawdown_answers.every(a => a !== null);
      case 2: // Volatility
        return form.volatility_answers.every(a => a !== null);
      case 5: // Constraints
        return constraintsOk;
      case 6: // Portfolio
        return portfolioOk;
      default:
        return true;
    }
  })();

  const setDrawdownAnswer = (idx, score) => {
    const next = [...form.drawdown_answers];
    next[idx] = score;
    setForm({ ...form, drawdown_answers: next });
  };
  const setVolatilityAnswer = (idx, score) => {
    const next = [...form.volatility_answers];
    next[idx] = score;
    setForm({ ...form, volatility_answers: next });
  };

  const ddScore = Math.round(form.drawdown_answers.filter(x => x !== null).reduce((a, b) => a + b, 0) / Math.max(1, form.drawdown_answers.filter(x => x !== null).length));
  const volScore = Math.round(form.volatility_answers.filter(x => x !== null).reduce((a, b) => a + b, 0) / Math.max(1, form.volatility_answers.filter(x => x !== null).length));

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
          drawdown_answers: form.drawdown_answers,
          volatility_answers: form.volatility_answers,
          drawdown_sensitivity: ddScore,
          volatility_sensitivity: volScore,
          goal_flexibility: form.goal_flexibility,
          concentration_pref: form.concentration_pref,
          start_cap: parseFloat(form.start_cap) || 100000,
          monthly_contrib: parseFloat(form.monthly_contrib) || 0,
          hard_constraints: form.hard_constraints
            .filter((c) => c.ticker && c.pct)
            .map((c) => ({ ticker: c.ticker.toUpperCase(), pct: parseFloat(c.pct) })),
          current_portfolio: form.current_portfolio
            .filter((c) => c.ticker && c.pct)
            .map((c) => ({ ticker: c.ticker.toUpperCase(), pct: parseFloat(c.pct) })),
        },
      };
      await api.post('/questionnaire/save', payload);
      
      // Update UI to show we're on the next step
      setSaving('Generating your customized portfolio...');
      
      // Auto-generate a recommended portfolio from the saved answers
      // This is the step that takes time (ML inference)
      const recommendRes = await api.post('/recommend');
      if (recommendRes.data?.portfolio?.id) {
        useStore.getState().setActivePortfolio(recommendRes.data.portfolio.id);
      }
      
      await fetchQuestionnaire();
      await fetchUserData();
      navigate('/');
    } catch (err) {
      console.error(err);
      alert('Failed to save questionnaire. Please check your inputs or try again later.');
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={labelStyle}>Starting Capital ($)</label>
                <input
                  type="number"
                  value={form.start_cap}
                  onChange={(e) => setForm({ ...form, start_cap: e.target.value })}
                  placeholder="100000"
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>Monthly Contribution ($)</label>
                <input
                  type="number"
                  value={form.monthly_contrib}
                  onChange={(e) => setForm({ ...form, monthly_contrib: e.target.value })}
                  placeholder="500"
                  style={inputStyle}
                />
              </div>
            </div>

            <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.05)' }} />

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
            {!stepOk && (
              <p style={{ color: '#f87171', fontSize: '0.75rem', marginTop: '1rem' }}>
                ✗ Please enter your starting capital and at least one financial goal with a valid amount and year.
              </p>
            )}
            </div>
          </div>
        );

      case 1: // Drawdown Tolerance
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Drawdown Tolerance
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              How do you handle market crashes? This scales your drawdown protection.
              <span style={{ color: '#60a5fa', fontWeight: 600 }}> Score: {ddScore}/10</span>
            </p>
            {DRAWDOWN_QUESTIONS.map((q, qi) => (
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
                    background: form.drawdown_answers[qi] === opt.score
                      ? 'rgba(59,130,246,0.12)' : 'transparent',
                    border: form.drawdown_answers[qi] === opt.score
                      ? '1px solid rgba(59,130,246,0.3)' : '1px solid transparent',
                    transition: 'all 0.15s',
                  }}>
                    <input
                      type="radio"
                      name={`dd-${qi}`}
                      checked={form.drawdown_answers[qi] === opt.score}
                      onChange={() => setDrawdownAnswer(qi, opt.score)}
                      style={{ accentColor: '#3b82f6' }}
                    />
                    <span style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>{opt.label}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        );

      case 2: // Volatility Tolerance
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Volatility Preference
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              Do you prefer smooth compounding or wild growth swings?
              <span style={{ color: '#60a5fa', fontWeight: 600 }}> Score: {volScore}/10</span>
            </p>
            {VOLATILITY_QUESTIONS.map((q, qi) => (
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
                    background: form.volatility_answers[qi] === opt.score
                      ? 'rgba(59,130,246,0.12)' : 'transparent',
                    border: form.volatility_answers[qi] === opt.score
                      ? '1px solid rgba(59,130,246,0.3)' : '1px solid transparent',
                    transition: 'all 0.15s',
                  }}>
                    <input
                      type="radio"
                      name={`vol-${qi}`}
                      checked={form.volatility_answers[qi] === opt.score}
                      onChange={() => setVolatilityAnswer(qi, opt.score)}
                      style={{ accentColor: '#3b82f6' }}
                    />
                    <span style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>{opt.label}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        );

      case 3: // Goal Flexibility
        const flexLabels = {
          1: "Must hit exactly",
          3: "Mostly strict",
          5: "Balanced",
          7: "Flexible",
          9: "Very lenient",
          10: "Return over Goal"
        };
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Goal Flexibility
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '2rem' }}>
              How satisfied would you be with a reasonable return even if your exact goal amount is missed?
            </p>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                display: 'inline-block', 
                padding: '0.2rem 0.6rem', 
                borderRadius: '1rem', 
                background: 'rgba(59, 130, 246, 0.2)', 
                color: '#60a5fa', 
                fontSize: '0.85rem', 
                fontWeight: 700,
                marginBottom: '1rem'
              }}>
                Value: {form.goal_flexibility}
              </div>
              <input
                type="range"
                min="1" max="10"
                value={form.goal_flexibility}
                onChange={(e) => setForm({ ...form, goal_flexibility: parseInt(e.target.value) })}
                style={{ width: '100%', accentColor: '#3b82f6' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 5px', color: '#475569', fontSize: '0.7rem', marginTop: '0.5rem' }}>
                {[1,2,3,4,5,6,7,8,9,10].map(v => <span key={v}>{v}</span>)}
              </div>
              
              <div style={{ 
                marginTop: '2rem', 
                minHeight: '4.5rem', 
                display: 'flex', 
                flexDirection: 'column',
                justifyContent: 'center',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '0.5rem',
                padding: '0.75rem',
                border: '1px solid rgba(255,255,255,0.05)',
                textAlign: 'left'
              }}>
                <span style={{ color: '#94a3b8', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
                  Interpretation
                </span>
                <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '1rem' }}>
                  {flexLabels[form.goal_flexibility] || "Balanced Sensitivity"}
                </span>
                <p style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                  {form.goal_flexibility <= 3 ? "Prioritizes meeting specific dollar targets over taking extra risk." : 
                   form.goal_flexibility >= 8 ? "Prioritizes total portfolio growth over rigid goal success." :
                   "Balances the need for growth with the requirement to hit targets."}
                </p>
              </div>
            </div>
          </div>
        );

      case 4: // Concentration Preference
        const concLabels = {
          1: "Max Conviction (1-2 stocks)",
          2: "Focused (3-5 stocks)",
          3: "Strategic (5-8 stocks)",
          4: "Active (8-12 stocks)",
          5: "Standard (12-15 stocks)",
          6: "Balanced (15-20 stocks)",
          7: "Diversified (20-25 stocks)",
          8: "Broad (25-30 stocks)",
          9: "Market Wide (30-40 stocks)",
          10: "Index Proxy (40+ stocks)"
        };
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Concentration Preference
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '2rem' }}>
              How many individual holdings do you prefer? Fewer holdings allow for higher conviction but more specific risk.
            </p>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                display: 'inline-block', 
                padding: '0.2rem 0.6rem', 
                borderRadius: '1rem', 
                background: 'rgba(167, 139, 250, 0.2)', 
                color: '#a78bfa', 
                fontSize: '0.85rem', 
                fontWeight: 700,
                marginBottom: '1rem'
              }}>
                Value: {form.concentration_pref}
              </div>
              <input
                type="range"
                min="1" max="10"
                value={form.concentration_pref}
                onChange={(e) => setForm({ ...form, concentration_pref: parseInt(e.target.value) })}
                style={{ width: '100%', accentColor: '#8b5cf6' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 5px', color: '#475569', fontSize: '0.7rem', marginTop: '0.5rem' }}>
                {[1,2,3,4,5,6,7,8,9,10].map(v => <span key={v}>{v}</span>)}
              </div>

              <div style={{ 
                marginTop: '2rem', 
                minHeight: '4.5rem', 
                display: 'flex', 
                flexDirection: 'column',
                justifyContent: 'center',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '0.5rem',
                padding: '0.75rem',
                border: '1px solid rgba(255,255,255,0.05)',
                textAlign: 'left'
              }}>
                <span style={{ color: '#94a3b8', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
                  Interpretation
                </span>
                <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '1rem' }}>
                  {concLabels[form.concentration_pref] || "Standard Diversification"}
                </span>
                <p style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                  {form.concentration_pref <= 3 ? "Optimizes for high-alpha concentrated bets on top-performing assets." : 
                   form.concentration_pref >= 8 ? "Optimizes for low-tracking error and broad market exposure." :
                   "Optimizes for a balanced mix of conviction and risk-mitigation."}
                </p>
              </div>
            </div>
          </div>
        );

      case 5: // Hard Constraints
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Specific Stock Preferences
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              Want to guarantee a position in specific companies? These will be carved out before optimization.
            </p>
            {form.hard_constraints.map((c, i) => (
              <div key={i} style={{ 
                display: 'grid', 
                gridTemplateColumns: '2fr 1fr auto', 
                gap: '0.5rem', 
                marginBottom: '0.75rem',
                alignItems: 'end'
              }}>
                <div>
                  <label style={labelStyle}>Ticker</label>
                  <TickerInput
                    value={c.ticker}
                    onChange={(v) => updateConstraint(i, 'ticker', v)}
                    onValidChange={(t, isValid) => setConstraintValid(i, t, isValid)}
                    placeholder="AAPL"
                    strict={true}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Allocation %</label>
                  <input style={inputStyle} type="number" value={c.pct} placeholder="10"
                    onChange={(e) => updateConstraint(i, 'pct', e.target.value)} />
                </div>
                <button onClick={() => removeConstraint(i)} style={{
                  background: 'none', border: 'none', color: '#f87171', cursor: 'pointer',
                  fontSize: '1.2rem', paddingBottom: '0.5rem',
                }}>×</button>
              </div>
            ))}
            {!constraintsOk && (
              <p style={{ color: '#f87171', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                ✗ One or more tickers are invalid or have 0% allocation. Please fix them before continuing.
              </p>
            )}
            <button onClick={addConstraint} style={{ ...btnSecondary, fontSize: '0.8rem', marginTop: '0.5rem' }}>
              + Add Ticker
            </button>
          </div>
        );

      case 6: // Current Portfolio
        return (
          <div>
            <h2 style={{ color: '#f1f5f9', fontWeight: 700, marginBottom: '0.5rem' }}>
              Current Portfolio (Optional)
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
              If you already hold investments, enter them here so we can compare and optimize.
            </p>
            {form.current_portfolio.map((c, i) => (
              <div key={i} style={{ 
                display: 'grid', 
                gridTemplateColumns: '2fr 1fr auto', 
                gap: '0.5rem', 
                marginBottom: '0.75rem',
                alignItems: 'end'
              }}>
                <div>
                  <label style={labelStyle}>Ticker</label>
                  <TickerInput
                    value={c.ticker}
                    onChange={(v) => updatePortfolio(i, 'ticker', v)}
                    onValidChange={(t, isValid) => setPortfolioValid(i, t, isValid)}
                    placeholder="VOO"
                    strict={true}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Allocation %</label>
                  <input style={inputStyle} type="number" value={c.pct} placeholder="50"
                    onChange={(e) => updatePortfolio(i, 'pct', e.target.value)} />
                </div>
                <button onClick={() => removePortfolioRow(i)} style={{
                  background: 'none', border: 'none', color: '#f87171', cursor: 'pointer',
                  fontSize: '1.2rem', paddingBottom: '0.5rem',
                }}>×</button>
              </div>
            ))}
            {!portfolioOk && (
              <p style={{ color: '#f87171', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                ✗ One or more tickers are invalid or have 0% allocation. Please fix or remove them.
              </p>
            )}
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
            <button
              onClick={() => setStep(step + 1)}
              disabled={!stepOk}
              style={{ ...btnPrimary, opacity: stepOk ? 1 : 0.45, cursor: stepOk ? 'pointer' : 'not-allowed' }}
            >
              Next
            </button>
          ) : (
            <button onClick={handleSave} disabled={saving || !stepOk} style={{
              ...btnPrimary, opacity: (saving || !stepOk) ? 0.5 : 1,
              cursor: (saving || !stepOk) ? 'not-allowed' : 'pointer',
            }}>
              {saving ? (typeof saving === 'string' ? saving : 'Saving...') : 'Complete & Generate'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
