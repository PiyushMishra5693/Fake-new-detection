"""
Command Line Interface for Fake News Detection App.
Usage:
    python cli.py "BOMBSHELL: Government caught installing microchips in tap water!"
    python cli.py --interactive
"""

import sys
import os
import argparse
import pandas as pd

# Ensure Windows terminal doesn't crash on emojis/special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import FakeNewsModelManager
from src.heuristics import analyze_linguistic_features

def load_or_train_model():
    data_path = os.path.join(os.path.dirname(__file__), "data", "news_dataset.csv")
    if not os.path.exists(data_path):
        from generate_dataset import build_dataset
        build_dataset()
        
    df = pd.read_csv(data_path)
    manager = FakeNewsModelManager(model_type="Ensemble")
    print(f"Training NLP models on {len(df)} curated samples...")
    metrics = manager.train_all(df)
    for model_name, m in metrics.items():
        print(f"  - {model_name:22} | Acc: {m['accuracy']*100:.1f}% | F1: {m['f1']:.3f}")
    return manager

def run_analysis(manager: FakeNewsModelManager, title: str, text: str = ""):
    print("\n" + "="*60)
    print(f"ANALYZING ARTICLE:\nTitle: {title}")
    if text:
        preview = text[:160] + "..." if len(text) > 160 else text
        print(f"Text:  {preview}")
    print("="*60)
    
    # 1. Model Prediction
    pred = manager.predict(title, text)
    verdict = pred["verdict"]
    confidence = pred["confidence"]
    badge = "[🚨 FAKE / UNRELIABLE]" if verdict == "FAKE" else "[✅ REAL / CREDIBLE]"
    
    print(f"\nPREDICTION RESULT: {badge}")
    print(f"Confidence:          {confidence:.1f}%")
    print(f"Fake Probability:    {pred['fake_probability_percent']:.1f}%")
    print(f"Real Probability:    {pred['real_probability_percent']:.1f}%")
    
    print("\nModel Consensus Breakdown:")
    for model_name, info in pred["model_breakdown"].items():
        print(f"  - {model_name:20}: {info['verdict']:4} (Fake: {info['fake_probability']*100:5.1f}%)")
        
    # 2. Linguistic Heuristics
    heuristics = analyze_linguistic_features(title, text)
    print(f"\nStylistic & Clickbait Indicators:")
    print(f"  - Sensationalism Score:     {heuristics['sensationalism_score']}/100 (Risk: {heuristics['risk_level']})")
    print(f"  - ALL-CAPS Words Ratio:     {heuristics['caps_ratio_percent']}% ({heuristics['caps_word_count']} words)")
    print(f"  - Exclamation Marks:        {heuristics['exclamation_count']}")
    if heuristics["flagged_clickbait_terms"]:
        print(f"  - Clickbait Triggers Found: {', '.join(heuristics['flagged_clickbait_terms'])}")
        
    # 3. Model Explainability Cues
    explanation = pred["explanation"]
    if explanation["fake_signals"]:
        cues = [f"{item['term']} (+{item['weight']})" for item in explanation["fake_signals"][:4]]
        print(f"\nTop Features Pushing 'FAKE': {', '.join(cues)}")
    if explanation["real_signals"]:
        cues = [f"{item['term']} (-{item['weight']})" for item in explanation["real_signals"][:4]]
        print(f"Top Features Pushing 'REAL': {', '.join(cues)}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Fake News Detector CLI")
    parser.add_argument("headline", nargs="?", default=None, help="News headline or text to evaluate")
    parser.add_argument("--body", default="", help="Optional full body text of the article")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")
    args = parser.parse_args()
    
    manager = load_or_train_model()
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                title = input("\n[Enter Headline]: ").strip()
                if title.lower() in ("quit", "exit", "q"):
                    break
                if not title:
                    continue
                body = input("[Enter Body (optional, press Enter to skip)]: ").strip()
                run_analysis(manager, title, body)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.headline:
        run_analysis(manager, args.headline, args.body)
    else:
        # Run a default real vs fake showcase
        print("\nRunning demonstration with standard real and fake test articles:")
        run_analysis(
            manager,
            "BOMBSHELL: Secret 5G Satellite Transmits Mind Control Waves Through Tap Water!",
            "Insiders confirm that clandestine orbital satellites are beaming classified scalar waves into municipal reservoirs. Share this immediately before tech monopolies delete it!"
        )
        run_analysis(
            manager,
            "NASA James Webb Space Telescope Detects Water Vapor in Rocky Planet Habitable Zone",
            "Astronomers using the James Webb Space Telescope have observed spectral signatures of atmospheric water vapor around an Earth-sized rocky exoplanet orbiting within its host star's habitable zone."
        )

if __name__ == "__main__":
    main()
