import React from 'react';

const fmt = (val) => {
  if (val === undefined || val === null) return '$0.00';
  return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

export default function BalanceStepTable({ steps, segments, portfolioName }) {
  if (!steps || steps.length === 0) {
    return (
      <div style={{
        padding: '1.5rem', textAlign: 'center', color: '#64748b',
        background: 'rgba(15,23,42,0.4)', borderRadius: '0.75rem',
        border: '1px solid rgba(255,255,255,0.05)'
      }}>
        No goal milestones recorded for this simulation.
      </div>
    );
  }

  const exportCSV = () => {
    const headers = ['Year', 'Remaining Balance', 'Top Allocation'];
    const rows = steps.map(s => {
      const seg = segments?.find(seg => seg.horizon_years?.[0] === s.year);
      const topAsset = seg ? Object.entries(seg.weights).sort((a,b) => b[1]-a[1])[0] : null;
      const allocStr = topAsset ? `${topAsset[0]} (${(topAsset[1]*100).toFixed(1)}%)` : 'Steady';
      return [s.year, s.balance.toFixed(2), allocStr];
    });
    const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `${portfolioName.replace(/\s+/g, '_')}_milestones.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{
      padding: '1.5rem', borderRadius: '0.75rem',
      background: 'rgba(15,23,42,0.6)', backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.06)',
      marginTop: '1.5rem'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <h3 style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem', margin: 0 }}>
          Goal Milestone Snapshots & Rebalancing
        </h3>
        <button 
          onClick={exportCSV}
          style={{
            background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)',
            color: '#60a5fa', padding: '0.4rem 0.8rem', borderRadius: '0.4rem',
            fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s'
          }}
          onMouseOver={e => e.currentTarget.style.background = 'rgba(59,130,246,0.2)'}
          onMouseOut={e => e.currentTarget.style.background = 'rgba(59,130,246,0.1)'}
        >
          Export CSV
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <th style={{ padding: '0.75rem 1rem', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Year</th>
              <th style={{ padding: '0.75rem 1rem', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Status</th>
              <th style={{ padding: '0.75rem 1rem', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Goal / Withdrawn</th>
              <th style={{ padding: '0.75rem 1rem', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Shortfall</th>
              <th style={{ padding: '0.75rem 1rem', color: '#64748b', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', textAlign: 'right' }}>Post-Cashout Balance</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step, idx) => {
              // Find the segment that STARTS at this milestone year
              const nextSegment = segments?.find(seg => seg.horizon_years?.[0] === step.year);
              const topAssets = nextSegment ? 
                Object.entries(nextSegment.weights)
                  .sort((a,b) => b[1]-a[1])
                  .slice(0, 2)
                  .map(([ticker, w]) => `${ticker} ${(w*100).toFixed(0)}%`)
                  .join(", ") : null;

              return (
                <tr key={idx} style={{ 
                  borderBottom: idx === steps.length - 1 ? 'none' : '1px solid rgba(255,255,255,0.05)',
                  transition: 'background 0.2s'
                }}
                onMouseOver={e => e.currentTarget.style.background = 'rgba(255,255,255,0.02)'}
                onMouseOut={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '1rem', color: '#f1f5f9', fontWeight: 600, fontSize: '0.9rem' }}>
                    Year {step.year}
                  </td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{ 
                      fontSize: '0.7rem', fontWeight: 600, padding: '0.2rem 0.5rem', borderRadius: '1rem',
                      background: step.balance >= 0 ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                      color: step.balance >= 0 ? '#4ade80' : '#fca5a5'
                    }}>
                      {step.balance >= 0 ? 'GOAL MET' : 'SHORTFALL'}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                      <span style={{ fontSize: '0.7rem' }}>Goal: {fmt(step.goal_amount)}</span>
                      <span style={{ color: step.shortfall > 0 ? '#fca5a5' : '#4ade80', fontWeight: 600 }}>
                        {fmt(step.withdrawn)}
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: '1rem', color: step.shortfall > 0 ? '#ef4444' : '#64748b', fontSize: '0.85rem', fontWeight: step.shortfall > 0 ? 700 : 400 }}>
                    {step.shortfall > 0 ? fmt(step.shortfall) : '—'}
                  </td>
                  <td style={{ padding: '1rem', color: '#60a5fa', fontWeight: 700, fontSize: '0.95rem', textAlign: 'right', letterSpacing: '-0.01em' }}>
                    {fmt(step.balance)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      <p style={{ color: '#475569', fontSize: '0.7rem', marginTop: '1.25rem', fontStyle: 'italic' }}>
        * Snapshots represent the average remaining portfolio value across 20 Monte Carlo paths after deducting that year's goal amount.
      </p>
    </div>
  );
}
