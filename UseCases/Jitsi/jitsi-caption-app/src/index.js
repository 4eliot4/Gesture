import React from 'react';
import ReactDOM from 'react-dom/client';
import { Routes, Route, BrowserRouter as Router } from 'react-router-dom';
import './index.css';
import App from './App';
import Playground from './Playground';
import Frames from './Frames';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  // <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/frames" element={<Frames />} />
      </Routes>
    </Router>
  // </React.StrictMode>
);

reportWebVitals();
