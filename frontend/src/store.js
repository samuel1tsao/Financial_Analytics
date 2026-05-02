import { create } from 'zustand';
import api from './api/client';

const API_BASE = 'http://localhost:8001/api/v1';

const useStore = create((set, get) => ({
  // ─── Auth State ─────────────────────────────────────────────────────────
  token: localStorage.getItem('token') || null,
  user: null,
  isAuthenticated: !!localStorage.getItem('token'),

  setToken: (token) => {
    localStorage.setItem('token', token);
    set({ token, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem('token');
    set({ token: null, user: null, isAuthenticated: false, portfolios: [], favorites: [] });
  },

  // ─── User Data ──────────────────────────────────────────────────────────
  portfolios: [],
  favorites: [],
  activePortfolioId: null,
  simCache: {}, // { portfolioId_goalsHash: simulationData }

  setActivePortfolio: (id) => set({ activePortfolioId: id }),

  fetchUserData: async () => {
    const { token } = get();
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/user/data`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Failed to fetch user data');
      const data = await res.json();
      set({
        user: data.user,
        portfolios: data.portfolios,
        favorites: data.favorites,
      });
    } catch (err) {
      console.error('fetchUserData error:', err);
    }
  },

  // ─── Questionnaire ─────────────────────────────────────────────────────
  questionnaire: null,
  hasCompletedQuestionnaire: false,

  fetchQuestionnaire: async () => {
    const { token } = get();
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/questionnaire/current`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (data.exists) {
        set({ questionnaire: data.answers, hasCompletedQuestionnaire: true });
      }
    } catch (err) {
      console.error('fetchQuestionnaire error:', err);
    }
  },

  setSimCache: (key, data) => set((state) => ({
    simCache: { ...state.simCache, [key]: data }
  })),

  deletePortfolio: async (id) => {
    const { fetchUserData } = get();
    console.log("Deleting portfolio:", id);
    try {
      const res = await api.delete(`/portfolio/${id}`);
      console.log("Delete response:", res.status, res.data);
      if (res.status === 200) {
        await fetchUserData();
        set({ activePortfolioId: null });
      }
    } catch (err) {
      console.error('deletePortfolio error:', err);
    }
  },

  // ─── Sidebar View ──────────────────────────────────────────────────────
  activeView: 'dashboard', // 'dashboard' | 'officials'
  setActiveView: (view) => set({ activeView: view }),
}));

export default useStore;
