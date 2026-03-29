import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import HomePage from './pages/HomePage';
import PredictPage from './pages/PredictPage';
import BreedExplorerPage from './pages/BreedExplorerPage';
import AboutPage from './pages/AboutPage';
import './App.css';

function App() {
  return (
    <Router>
      <nav className="navbar">
        <div className="navbar-inner">
          <NavLink to="/" className="navbar-brand">
            🐄 <span>CattleAI</span>
          </NavLink>
          <ul className="nav-links">
            <li><NavLink to="/" end>Home</NavLink></li>
            <li><NavLink to="/predict">Predict</NavLink></li>
            <li><NavLink to="/breeds">Breeds</NavLink></li>
            <li><NavLink to="/about">About</NavLink></li>
          </ul>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/breeds" element={<BreedExplorerPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>

      <footer className="footer">
        <p>
          Built by <a href="https://www.linkedin.com/in/sajit9285/" target="_blank" rel="noreferrer">Ajit Kumar Singh</a>
          {' '} · Powered by PyTorch & FastAPI · {new Date().getFullYear()}
        </p>
      </footer>
    </Router>
  );
}

export default App;
