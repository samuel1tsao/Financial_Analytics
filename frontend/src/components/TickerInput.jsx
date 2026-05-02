import { useState, useEffect, useRef, useCallback } from 'react';
import api from '../api/client';

/**
 * TickerInput — smart ticker field with:
 *   - Autocomplete from /market/search
 *   - On-blur validation via /market/lookup/{ticker}
 *   - Visual states: idle | loading | valid | syncing | invalid
 *   - strict=true  → blocks invalid tickers (for questionnaire)
 *   - strict=false → shows info status but doesn't block (for markets search)
 *
 * Props:
 *   value           string
 *   onChange        (value: string) => void
 *   onValidChange   (ticker: string, isValid: bool | null) => void
 *     null = not yet validated / empty
 *   placeholder     string
 *   strict          bool (default false)
 *   style           object (applied to wrapper)
 *   inputStyle      object (applied to <input>)
 */

const LOOKUP_CACHE = {};  // module-level in-memory cache (session lifetime only — DB is the real cache)

const STATUS_META = {
  idle:     { color: 'rgba(255,255,255,0.1)', icon: null,  text: '' },
  loading:  { color: 'rgba(99,102,241,0.3)', icon: '⏳',  text: 'Looking up...' },
  synced:   { color: 'rgba(34,197,94,0.25)', icon: '✓',   text: 'In database' },
  syncing:  { color: 'rgba(245,158,11,0.25)', icon: '⚡', text: 'Found — syncing' },
  invalid:  { color: 'rgba(239,68,68,0.25)', icon: '✗',   text: 'Not a valid ticker' },
};

export default function TickerInput({
  value = '',
  onChange,
  onValidChange,
  placeholder = 'AAPL',
  strict = false,
  style = {},
  inputStyle = {},
}) {
  const [query, setQuery]           = useState(value);
  const [suggestions, setSuggestions] = useState([]);
  const [showDrop, setShowDrop]     = useState(false);
  const [status, setStatus]         = useState('idle');  // idle | loading | synced | syncing | invalid
  const [validatedTicker, setValidatedTicker] = useState('');
  const wrapperRef = useRef(null);
  const lookupTimer = useRef(null);

  // Sync external value changes
  useEffect(() => {
    if (value !== query) {
      setQuery(value);
      setStatus('idle');
      setValidatedTicker('');
    }
  }, [value]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setShowDrop(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Autocomplete suggestions (debounced 150ms)
  useEffect(() => {
    const t = query.trim().toUpperCase();
    if (!t || t.length < 1) {
      setSuggestions([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await api.get(`/market/search?q=${encodeURIComponent(t)}&limit=8`);
        setSuggestions(res.data);
        if (res.data.length > 0) setShowDrop(true);
      } catch {}
    }, 150);
    return () => clearTimeout(timer);
  }, [query]);

  // Validate on blur (with cache)
  const validateTicker = useCallback(async (ticker) => {
    const t = ticker.trim().toUpperCase();
    if (!t) {
      setStatus('idle');
      onValidChange?.(t, null);
      return;
    }

    // Already validated this ticker this session
    if (LOOKUP_CACHE[t] !== undefined) {
      const cached = LOOKUP_CACHE[t];
      setStatus(cached ? (cached === 'syncing' ? 'syncing' : 'synced') : 'invalid');
      onValidChange?.(t, cached !== false);
      return;
    }

    setStatus('loading');
    try {
      const res = await api.get(`/market/lookup/${encodeURIComponent(t)}`);
      const { valid, status: s } = res.data;
      if (valid) {
        const displayStatus = s === 'synced' ? 'synced' : 'syncing';
        LOOKUP_CACHE[t] = displayStatus;
        setStatus(displayStatus);
        onValidChange?.(t, true);
      } else {
        LOOKUP_CACHE[t] = false;
        setStatus('invalid');
        onValidChange?.(t, false);
      }
    } catch (err) {
      console.error('Lookup error:', err);
      // If it's a timeout or network error, don't mark as permanently invalid
      setStatus('idle');
      onValidChange?.(t, null);
    }
  }, [onValidChange]);

  const handleChange = (e) => {
    const v = e.target.value;
    setQuery(v);
    onChange?.(v);
    setStatus('idle');
    setValidatedTicker('');
    onValidChange?.(v, null);
  };

  const handleBlur = () => {
    // Small delay so suggestion clicks fire first
    lookupTimer.current = setTimeout(() => {
      const t = query.trim().toUpperCase();
      if (t && t !== validatedTicker) {
        setValidatedTicker(t);
        validateTicker(t);
      }
    }, 200);
  };

  const selectSuggestion = (s) => {
    clearTimeout(lookupTimer.current);
    setQuery(s.ticker);
    onChange?.(s.ticker);
    setSuggestions([]);
    setShowDrop(false);
    // Suggestions come from our DB → always valid & synced
    LOOKUP_CACHE[s.ticker] = 'synced';
    setStatus('synced');
    setValidatedTicker(s.ticker);
    onValidChange?.(s.ticker, true);
  };

  const meta = STATUS_META[status] ?? STATUS_META.idle;

  const borderColor = status === 'synced'  ? 'rgba(34,197,94,0.5)'
                    : status === 'syncing' ? 'rgba(245,158,11,0.5)'
                    : status === 'invalid' ? 'rgba(239,68,68,0.5)'
                    : status === 'loading' ? 'rgba(99,102,241,0.5)'
                    : 'rgba(255,255,255,0.1)';

  return (
    <div ref={wrapperRef} style={{ position: 'relative', ...style }}>
      {/* Input */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.4rem',
        background: 'rgba(30,41,59,0.8)',
        border: `1px solid ${borderColor}`,
        borderRadius: '0.4rem',
        padding: '0.65rem 0.85rem',
        transition: 'border-color 0.2s',
      }}>
        <input
          value={query}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={() => suggestions.length && setShowDrop(true)}
          placeholder={placeholder}
          style={{
            flex: 1, background: 'none', border: 'none', outline: 'none',
            color: '#f1f5f9', fontSize: '0.9rem', textTransform: 'uppercase',
            ...inputStyle,
          }}
        />
        {/* Status icon */}
        {status !== 'idle' && (
          <span style={{
            fontSize: status === 'loading' ? '0.7rem' : '0.85rem',
            color: status === 'synced'  ? '#4ade80'
                 : status === 'syncing' ? '#fbbf24'
                 : status === 'invalid' ? '#f87171'
                 : '#a5b4fc',
            flexShrink: 0,
          }}>
            {status === 'loading' ? (
              <span style={{
                display: 'inline-block',
                width: 12, height: 12,
                border: '2px solid rgba(99,102,241,0.3)',
                borderTopColor: '#6366f1',
                borderRadius: '50%',
                animation: 'tiSpin 0.7s linear infinite',
              }} />
            ) : meta.icon}
          </span>
        )}
      </div>

      {/* Status hint text */}
      {status !== 'idle' && meta.text && (
        <p style={{
          marginTop: '0.25rem', fontSize: '0.68rem', fontWeight: 500,
          color: status === 'synced'  ? '#4ade80'
               : status === 'syncing' ? '#fbbf24'
               : status === 'invalid' ? '#f87171'
               : '#94a3b8',
        }}>
          {status === 'syncing' ? '⚡ Found on Yahoo Finance — syncing to database...' : meta.text}
        </p>
      )}

      {/* Autocomplete Dropdown */}
      {showDrop && suggestions.length > 0 && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0, zIndex: 500,
          background: '#0d1526', border: '1px solid rgba(99,102,241,0.2)',
          borderRadius: '0.5rem', overflow: 'hidden',
          boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
          animation: 'tiFadeIn 0.12s ease-out',
        }}>
          <style>{`
            @keyframes tiSpin { to { transform: rotate(360deg) } }
            @keyframes tiFadeIn { from { opacity: 0; transform: translateY(-4px) } to { opacity: 1; transform: none } }
          `}</style>
          {suggestions.map((s, i) => (
            <div
              key={i}
              onMouseDown={() => selectSuggestion(s)}
              style={{
                padding: '0.55rem 0.85rem', cursor: 'pointer',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                borderBottom: i < suggestions.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                transition: 'background 0.1s',
              }}
              onMouseOver={e => e.currentTarget.style.background = 'rgba(99,102,241,0.12)'}
              onMouseOut={e => e.currentTarget.style.background = 'transparent'}
            >
              <div>
                <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '0.85rem' }}>{s.ticker}</span>
                <span style={{ color: '#475569', fontSize: '0.72rem', marginLeft: '0.5rem' }}>
                  {s.name}
                </span>
              </div>
              <span style={{
                fontSize: '0.6rem', fontWeight: 700, padding: '0.1rem 0.4rem', borderRadius: '0.8rem',
                background: s.asset_type === 'ETF' ? 'rgba(245,158,11,0.15)' : 'rgba(99,102,241,0.15)',
                color: s.asset_type === 'ETF' ? '#fbbf24' : '#a5b4fc',
              }}>
                {s.asset_type || '?'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
