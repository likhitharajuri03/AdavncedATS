// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Import components
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import JobSeekerDashboard from './pages/JobSeekerDashboard';
import RecruiterDashboard from './pages/RecruiterDashboard';
import ResumeAnalyzer from './components/ResumeAnalyzer';
import JobMatcher from './components/JobMatcher';
import ResumeBuilder from './components/ResumeBuilder';
import VirtualAssistant from './components/VirtualAssistant';

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    // Check if user is logged in
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');
    
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const handleLogin = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
    localStorage.setItem('token', authToken);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route 
            path="/login" 
            element={!token ? <Login onLogin={handleLogin} /> : <Navigate to="/dashboard" />} 
          />
          <Route 
            path="/register" 
            element={!token ? <Register onRegister={handleLogin} /> : <Navigate to="/dashboard" />} 
          />
          
          <Route 
            path="/dashboard" 
            element={
              token ? (
                user?.user_type === 'job_seeker' ? (
                  <JobSeekerDashboard user={user} onLogout={handleLogout} token={token} />
                ) : (
                  <RecruiterDashboard user={user} onLogout={handleLogout} token={token} />
                )
              ) : (
                <Navigate to="/login" />
              )
            } 
          />
          
          <Route 
            path="/analyze" 
            element={token ? <ResumeAnalyzer token={token} /> : <Navigate to="/login" />} 
          />
          
          <Route 
            path="/jobs" 
            element={token ? <JobMatcher token={token} /> : <Navigate to="/login" />} 
          />
          
          <Route 
            path="/builder" 
            element={token ? <ResumeBuilder token={token} /> : <Navigate to="/login" />} 
          />
          
          <Route 
            path="/assistant" 
            element={token ? <VirtualAssistant token={token} /> : <Navigate to="/login" />} 
          />
          
          <Route path="/" element={<Navigate to="/dashboard" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;


// (removed duplicate component; ResumeAnalyzer is now in its own file: ./components/ResumeAnalyzer.jsx)