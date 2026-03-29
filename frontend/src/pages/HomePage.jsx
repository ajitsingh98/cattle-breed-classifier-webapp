import { Link } from 'react-router-dom';

function HomePage() {
    return (
        <div className="page">
            <section className="hero">
                <h1>
                    Identify Your <span className="accent">Cattle Breed</span> in Seconds
                </h1>
                <p>
                    AI-powered breed classification for 26 indigenous Indian cattle and buffalo breeds.
                    Upload a photo, point your camera, or paste a URL — get instant results.
                </p>
                <Link to="/predict">
                    <button className="hero-cta">
                        🔍 Start Classifying
                    </button>
                </Link>
            </section>

            <section className="features-grid">
                <div className="card feature-card">
                    <div className="feature-icon">📸</div>
                    <h3>Multiple Input Modes</h3>
                    <p>Upload images, capture from camera, or paste a URL. Works on any device with a browser.</p>
                </div>
                <div className="card feature-card">
                    <div className="feature-icon">🧠</div>
                    <h3>Deep Learning Models</h3>
                    <p>Trained and compared 4 architectures — MLP, CNN, ResNet50, and Vision Transformer — to find the best.</p>
                </div>
                <div className="card feature-card">
                    <div className="feature-icon">📋</div>
                    <h3>Breed Information</h3>
                    <p>Get detailed metadata: region, milk yield, lifespan, primary use, and physical characteristics.</p>
                </div>
                <div className="card feature-card">
                    <div className="feature-icon">🌾</div>
                    <h3>Farmer-Friendly</h3>
                    <p>Designed for real-world use. Clear confidence indicators, image quality tips, and easy navigation.</p>
                </div>
                <div className="card feature-card">
                    <div className="feature-icon">⚡</div>
                    <h3>Fast & Reliable</h3>
                    <p>Sub-second predictions with confidence scores. Know when to trust the result.</p>
                </div>
                <div className="card feature-card">
                    <div className="feature-icon">🐃</div>
                    <h3>26 Breeds</h3>
                    <p>Covers major indigenous cow and buffalo breeds from across India with verified metadata.</p>
                </div>
            </section>
        </div>
    );
}

export default HomePage;
