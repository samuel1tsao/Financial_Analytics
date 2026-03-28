import { useState } from 'react';
import useStore from '../store';

export default function LeftSidebar() {
  const { portfolios, favorites, activePortfolioId, setActivePortfolio } = useStore();
  const [manageMode, setManageMode] = useState(false);
  const [selected, setSelected] = useState(new Set());

  const toggleSelect = (id) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const handleDelete = async () => {
    if (selected.size === 0) return;
    try {
      const token = localStorage.getItem('token');
      await fetch('http://localhost:8000/api/v1/profile', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ ids: Array.from(selected) }),
      });
      setSelected(new Set());
      setManageMode(false);
      useStore.getState().fetchUserData();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <aside style={{
      width: 240,
      minHeight: 'calc(100vh - 56px)',
      background: 'rgba(15, 23, 42, 0.6)',
      borderRight: '1px solid rgba(255,255,255,0.06)',
      padding: '1rem 0',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Section: My Portfolios */}
      <div style={{ padding: '0 1rem', marginBottom: '0.5rem' }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: '0.75rem',
        }}>
          <span style={{
            fontSize: '0.7rem', fontWeight: 600, color: '#64748b',
            textTransform: 'uppercase', letterSpacing: '0.08em',
          }}>
            Portfolios
          </span>
          <button
            onClick={() => { setManageMode(!manageMode); setSelected(new Set()); }}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: '#64748b', fontSize: '0.7rem',
            }}
          >
            {manageMode ? 'Done' : 'Manage'}
          </button>
        </div>

        {portfolios.length === 0 ? (
          <p style={{ color: '#475569', fontSize: '0.8rem', fontStyle: 'italic' }}>
            No portfolios yet
          </p>
        ) : (
          portfolios.map((p) => (
            <div
              key={p.id}
              onClick={() => !manageMode && setActivePortfolio(p.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.55rem 0.75rem',
                borderRadius: '0.5rem',
                marginBottom: '0.25rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
                background: activePortfolioId === p.id
                  ? 'rgba(59,130,246,0.15)'
                  : 'transparent',
                border: activePortfolioId === p.id
                  ? '1px solid rgba(59,130,246,0.25)'
                  : '1px solid transparent',
              }}
            >
              {manageMode && (
                <input
                  type="checkbox"
                  checked={selected.has(p.id)}
                  onChange={() => toggleSelect(p.id)}
                  style={{ accentColor: '#ef4444' }}
                />
              )}
              <span style={{
                fontSize: '0.85rem',
                color: activePortfolioId === p.id ? '#60a5fa' : '#cbd5e1',
                fontWeight: activePortfolioId === p.id ? 600 : 400,
              }}>
                {p.profile_name}
              </span>
              {p.is_current && (
                <span style={{
                  fontSize: '0.6rem', background: 'rgba(34,197,94,0.15)',
                  color: '#4ade80', padding: '0.1rem 0.4rem',
                  borderRadius: '1rem', fontWeight: 600,
                }}>
                  CURRENT
                </span>
              )}
            </div>
          ))
        )}

        {manageMode && selected.size > 0 && (
          <button
            onClick={handleDelete}
            style={{
              marginTop: '0.5rem',
              width: '100%',
              padding: '0.5rem',
              borderRadius: '0.4rem',
              background: 'rgba(239,68,68,0.15)',
              border: '1px solid rgba(239,68,68,0.3)',
              color: '#f87171',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: 600,
            }}
          >
            Delete {selected.size} profile{selected.size > 1 ? 's' : ''}
          </button>
        )}
      </div>

      {/* Divider */}
      <div style={{
        height: 1, background: 'rgba(255,255,255,0.06)',
        margin: '0.75rem 1rem',
      }} />

      {/* Section: Favorited Officials */}
      <div style={{ padding: '0 1rem' }}>
        <span style={{
          fontSize: '0.7rem', fontWeight: 600, color: '#64748b',
          textTransform: 'uppercase', letterSpacing: '0.08em',
          display: 'block', marginBottom: '0.75rem',
        }}>
          Tracked Officials
        </span>
        {favorites.length === 0 ? (
          <p style={{ color: '#475569', fontSize: '0.8rem', fontStyle: 'italic' }}>
            No favorites yet
          </p>
        ) : (
          favorites.map((officialId) => (
            <div
              key={officialId}
              style={{
                padding: '0.55rem 0.75rem',
                borderRadius: '0.5rem',
                marginBottom: '0.25rem',
                cursor: 'pointer',
                color: '#cbd5e1',
                fontSize: '0.85rem',
                transition: 'background 0.15s',
              }}
              onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
            >
              ⭐ {officialId}
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
