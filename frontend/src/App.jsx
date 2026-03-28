import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import useStore from './store';
import AuthPage from './views/Auth/Login';
import TopNavigation from './components/TopNavigation';
import LeftSidebar from './components/LeftSidebar';
import DashboardMain from './views/Dashboard/Main';
import QuestionnaireStepper from './views/Questionnaire/Stepper';
import Directory from './views/Officials/Directory';

function ProtectedLayout() {
  const { fetchUserData, fetchQuestionnaire, activeView } = useStore();

  useEffect(() => {
    fetchUserData();
    fetchQuestionnaire();
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0a0e1a' }}>
      <TopNavigation />
      <div style={{ display: 'flex' }}>
        <LeftSidebar />
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {activeView === 'dashboard' && <DashboardMain />}
          {activeView === 'officials' && <Directory />}
        </main>
      </div>
    </div>
  );
}

function PrivateRoute({ children }) {
  const isAuthenticated = useStore((s) => s.isAuthenticated);
  return isAuthenticated ? children : <Navigate to="/login" />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<AuthPage />} />
        <Route
          path="/questionnaire"
          element={
            <PrivateRoute>
              <QuestionnaireStepper />
            </PrivateRoute>
          }
        />
        <Route
          path="/"
          element={
            <PrivateRoute>
              <ProtectedLayout />
            </PrivateRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
