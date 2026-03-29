function AboutPage() {
    return (
        <div className="page">
            <h2 className="section-title">About CattleAI</h2>
            <p className="section-subtitle">
                An AI-powered cattle breed classification system for Indian indigenous breeds.
            </p>

            <div className="about-grid">
                <div className="card about-card">
                    <h3>🧠 The Models</h3>
                    <p>Four deep learning approaches were trained and rigorously compared:</p>
                    <ul>
                        <li>MLP Baseline — flatten + dense layers</li>
                        <li>CNN from Scratch — 5 conv blocks + GAP</li>
                        <li>ResNet50 Transfer Learning — ImageNet pretrained</li>
                        <li>ViT-B/16 Transfer Learning — Vision Transformer</li>
                    </ul>
                    <p style={{ marginTop: '0.75rem' }}>
                        The best model is selected using a weighted composite score considering F1, accuracy,
                        inference speed, and model size.
                    </p>
                </div>

                <div className="card about-card">
                    <h3>📊 The Dataset</h3>
                    <p>
                        3,056 images across 26 indigenous Indian breeds (21 cow + 5 buffalo breeds).
                        Stratified 70/15/15 train/val/test split.
                    </p>
                    <ul>
                        <li>Images resized to 224×224 pixels</li>
                        <li>Augmentation: flip, rotation, jitter, crop</li>
                        <li>ImageNet normalization applied</li>
                        <li>Corrupt image validation at preprocessing</li>
                    </ul>
                </div>

                <div className="card about-card">
                    <h3>⚙️ Tech Stack</h3>
                    <ul>
                        <li>PyTorch 2.x + torchvision + timm</li>
                        <li>FastAPI backend with Pydantic schemas</li>
                        <li>React + Vite frontend</li>
                        <li>Docker containerized deployment</li>
                        <li>Config-driven experiments with YAML</li>
                    </ul>
                </div>

                <div className="card about-card">
                    <h3>🎯 Best Model Selection</h3>
                    <p>Weighted scoring ensures the production model balances performance and practicality:</p>
                    <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                        <div className="breed-card-detail"><span className="label">Macro F1</span><span className="value">50%</span></div>
                        <div className="breed-card-detail"><span className="label">Top-1 Accuracy</span><span className="value">20%</span></div>
                        <div className="breed-card-detail"><span className="label">Inference Latency</span><span className="value">15%</span></div>
                        <div className="breed-card-detail"><span className="label">Model Size</span><span className="value">10%</span></div>
                        <div className="breed-card-detail"><span className="label">Calibration</span><span className="value">5%</span></div>
                    </div>
                </div>

                <div className="card about-card">
                    <h3>🌾 For Farmers</h3>
                    <p>
                        This tool is designed for real-world agricultural use. Features include:
                    </p>
                    <ul>
                        <li>Camera capture for field use</li>
                        <li>Low-confidence warnings for uncertain predictions</li>
                        <li>Image quality tips for better results</li>
                        <li>Breed details including milk yield and primary use</li>
                        <li>Works offline after initial load (PWA-ready)</li>
                    </ul>
                </div>

                <div className="card about-card">
                    <h3>👤 Creator</h3>
                    <p>
                        Built by <a href="https://www.linkedin.com/in/sajit9285/" target="_blank" rel="noreferrer">
                            Ajit Kumar Singh</a>.
                    </p>
                    <p style={{ marginTop: '0.5rem' }}>
                        Source code on{' '}
                        <a href="https://github.com/sajit9285/cattle-breed-classifier-webapp" target="_blank" rel="noreferrer">
                            GitHub
                        </a>.
                    </p>
                    <ul style={{ marginTop: '0.5rem' }}>
                        <li>Backend: FastAPI with PyTorch inference</li>
                        <li>Frontend: React + Vite</li>
                        <li>Training: Jupyter notebooks with shared ML package</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}

export default AboutPage;
