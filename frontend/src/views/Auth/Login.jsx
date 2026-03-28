import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import useStore from '../../store';
import api from '../../api/client';

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const setToken = useStore((s) => s.setToken);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        const res = await api.post('/user/login', { email, password });
        setToken(res.data.access_token);
      } else {
        await api.post('/user/register', { email, password });
        const res = await api.post('/user/login', { email, password });
        setToken(res.data.access_token);
      }
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%)',
    }}>
      <div style={{
        width: '100%',
        maxWidth: 420,
        padding: '2.5rem',
        borderRadius: '1rem',
        background: 'rgba(17, 24, 39, 0.8)',
        backdropFilter: 'blur(24px)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)',
      }}>
        {/* Logo & Title */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            width: 56, height: 56, margin: '0 auto 1rem',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.5rem', fontWeight: 700, color: '#fff',
            boxShadow: '0 0 30px rgba(59,130,246,0.3)',
          }}>
            $
          </div>
          <h1 style={{
            fontSize: '1.5rem', fontWeight: 700, color: '#f1f5f9',
            margin: 0, letterSpacing: '-0.02em',
          }}>
            Stock Recommender
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '0.875rem', marginTop: '0.5rem' }}>
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </p>
        </div>

        {/* Error Banner */}
        {error && (
          <div style={{
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            borderRadius: '0.5rem',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            color: '#fca5a5',
            fontSize: '0.875rem',
          }}>
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8', fontSize: '0.8rem', fontWeight: 500 }}>
            Email
          </label>
          <input
            id="auth-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={{
              width: '100%', padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              background: 'rgba(30, 41, 59, 0.8)',
              border: '1px solid rgba(255,255,255,0.1)',
              color: '#f1f5f9', fontSize: '0.95rem',
              outline: 'none', boxSizing: 'border-box',
              marginBottom: '1rem',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
            onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
            placeholder="you@example.com"
          />

          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8', fontSize: '0.8rem', fontWeight: 500 }}>
            Password
          </label>
          <input
            id="auth-password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{
              width: '100%', padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              background: 'rgba(30, 41, 59, 0.8)',
              border: '1px solid rgba(255,255,255,0.1)',
              color: '#f1f5f9', fontSize: '0.95rem',
              outline: 'none', boxSizing: 'border-box',
              marginBottom: '1.5rem',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
            onBlur={(e) => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
            placeholder="••••••••"
          />

          <button
            id="auth-submit"
            type="submit"
            disabled={loading}
            style={{
              width: '100%', padding: '0.85rem',
              borderRadius: '0.5rem',
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: '#fff', fontWeight: 600, fontSize: '0.95rem',
              border: 'none', cursor: 'pointer',
              transition: 'opacity 0.2s, transform 0.2s',
              opacity: loading ? 0.7 : 1,
              boxShadow: '0 4px 15px rgba(59,130,246,0.3)',
            }}
            onMouseOver={(e) => { if (!loading) e.target.style.transform = 'translateY(-1px)'; }}
            onMouseOut={(e) => { e.target.style.transform = 'translateY(0)'; }}
          >
            {loading ? 'Loading...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        {/* Toggle */}
        <p style={{ textAlign: 'center', marginTop: '1.5rem', color: '#94a3b8', fontSize: '0.85rem' }}>
          {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
          <button
            onClick={() => { setIsLogin(!isLogin); setError(''); }}
            style={{
              background: 'none', border: 'none',
              color: '#60a5fa', cursor: 'pointer',
              fontWeight: 600, fontSize: '0.85rem',
              textDecoration: 'underline',
            }}
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  );
}
