import os
import re
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def perform_eda():
    """Analyze the historical knowledge base."""
    print("--- EXPLORATORY DATA ANALYSIS (EDA) ---")
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    stats = {}
    all_text = ""
    
    print(f"\nTotal Files: {len(files)}")
    
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            all_text += content + " "
            
            words = content.split()
            stats[filename] = {
                "word_count": len(words),
                "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
            }

    # Document Stats
    print("\nDocument Statistics:")
    for doc, s in stats.items():
        print(f"- {doc}: {s['word_count']} words (Avg length: {s['avg_word_length']})")

    # Keyword Analysis (Simple Feature Engineering)
    stop_words = set(["the", "and", "a", "of", "to", "in", "is", "it", "that", "was", "for", "on", "with", "as", "at"])
    clean_words = [w.lower().strip(".,()!?:;") for w in all_text.split() if w.lower().strip(".,()!?:;") not in stop_words and len(w) > 3]
    top_keywords = Counter(clean_words).most_common(10)
    
    print("\nTop 10 Keywords (Feature Importance):")
    for word, count in top_keywords:
        print(f"- {word}: {count} occurrences")

    # Era Detection (EDA Insights)
    eras = {
        "Ancient": len(re.findall(r"BC|BCE|Ancient", all_text, re.IGNORECASE)),
        "Medieval": len(re.findall(r"Medieval|Knights|Crusades|Castle", all_text, re.IGNORECASE)),
        "Modern": len(re.findall(r"World War|19|20|Modern", all_text, re.IGNORECASE))
    }
    
    print("\nKnowledge Distribution by Era:")
    for era, count in eras.items():
        print(f"- {era}: {count} relevant tokens found")

    print("\n--- EDA COMPLETE ---")

if __name__ == "__main__":
    perform_eda()
