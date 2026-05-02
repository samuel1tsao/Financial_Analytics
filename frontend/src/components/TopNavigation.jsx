import useStore from '../store';

export default function TopNavigation() {
  const { activeView, setActiveView, user, logout } = useStore();

  const tabs = [
    { id: 'dashboard', label: 'My Portfolios' },
    { id: 'markets', label: '📈 Markets' },
    { id: 'officials', label: 'Public Officials' },
  ];

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 1.5rem',
      height: 56,
      background: 'rgba(15, 23, 42, 0.85)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(255,255,255,0.06)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{
          width: 32, height: 32, borderRadius: '50%',
          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '0.85rem', fontWeight: 700, color: '#fff',
        }}>$</div>
        <span style={{
          fontWeight: 700, fontSize: '1rem', color: '#f1f5f9',
          letterSpacing: '-0.02em',
        }}>StockRec</span>
      </div>

      {/* Tabs */}
      <nav style={{ display: 'flex', gap: '0.25rem' }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            id={`nav-tab-${tab.id}`}
            onClick={() => setActiveView(tab.id)}
            style={{
              padding: '0.5rem 1.25rem',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '0.85rem',
              transition: 'all 0.2s ease',
              background: activeView === tab.id
                ? 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2))'
                : 'transparent',
              color: activeView === tab.id ? '#60a5fa' : '#94a3b8',
              border: activeView === tab.id
                ? '1px solid rgba(59,130,246,0.3)'
                : '1px solid transparent',
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* User / Logout */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {user && (
          <span style={{ color: '#64748b', fontSize: '0.8rem' }}>
            {user.email}
          </span>
        )}
        <button
          id="logout-btn"
          onClick={logout}
          style={{
            padding: '0.4rem 1rem',
            borderRadius: '0.4rem',
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.2)',
            color: '#f87171',
            cursor: 'pointer',
            fontSize: '0.8rem',
            fontWeight: 500,
            transition: 'all 0.2s',
          }}
        >
          Logout
        </button>
      </div>
    </header>
  );
}
