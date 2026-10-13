"""
download_data.py — Expands the knowledge base by downloading articles from Wikipedia.
"""

import os
import sys
import wikipedia

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Extended list of historical topics to bulk up the vectorstore
topics = [
    "World War II",
    "Roman Empire",
    "Industrial Revolution",
    "French Revolution",
    "Ancient Egypt",
    "American Civil War",
    "Cold War",
    "Renaissance",
    "Ottoman Empire",
    "Ming Dynasty",
    "Aztec Empire",
    "Vedic Period",
    "Gupta Empire",
    "British Raj",
    "Age of Discovery"
]

print("🌍 Downloading external datasets from Wikipedia...")

found_count = 0
for topic in topics:
    filename = os.path.join(data_dir, f"{topic.lower().replace(' ', '_')}.txt")
    
    # Skip if we already downloaded it recently
    if os.path.exists(filename):
        print(f"⏭️ Skipping '{topic}': file already exists.")
        continue

    try:
        # Auto-suggest False ensures we don't accidentally get an irrelevant disambiguation
        page = wikipedia.page(topic, auto_suggest=False)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(page.content)
        print(f"✅ Downloaded: {topic} ({len(page.content)} characters)")
        found_count += 1
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Disambiguation for {topic}. Skipping.")
    except Exception as e:
        print(f"❌ Failed to download {topic}: {str(e)}")

print(f"\n🎉 Successfully added {found_count} new historical articles to the dataset!")
