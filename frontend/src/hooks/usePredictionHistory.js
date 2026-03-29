import { useState, useCallback } from 'react';

const HISTORY_KEY = 'cattle_classifier_history';
const MAX_HISTORY = 20;

export function usePredictionHistory() {
    const [history, setHistory] = useState(() => {
        try {
            const stored = localStorage.getItem(HISTORY_KEY);
            return stored ? JSON.parse(stored) : [];
        } catch {
            return [];
        }
    });

    const addPrediction = useCallback((prediction, imagePreview) => {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            predictedBreed: prediction.predicted_breed,
            confidence: prediction.confidence,
            topK: prediction.top_k,
            imagePreview: imagePreview?.substring(0, 200), // Truncate for storage
        };

        setHistory(prev => {
            const updated = [entry, ...prev].slice(0, MAX_HISTORY);
            try {
                localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
            } catch { /* Storage full */ }
            return updated;
        });
    }, []);

    const clearHistory = useCallback(() => {
        setHistory([]);
        localStorage.removeItem(HISTORY_KEY);
    }, []);

    return { history, addPrediction, clearHistory };
}
