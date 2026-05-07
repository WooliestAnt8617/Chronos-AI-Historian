import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import os

DATA = [
    #  Ancient History
    ("Who was Julius Caesar?", "Ancient"), ("The fall of the Roman Empire", "Ancient"),
    ("Who founded Rome?", "Ancient"), ("Ancient Greek philosophy and Socrates", "Ancient"),
    ("The construction of the Pyramids", "Ancient"), ("Vikings and Norse mythology", "Ancient"),
    ("Alexander the Great's conquests", "Ancient"), ("The Punic Wars and Carthage", "Ancient"),
    ("Cleopatra and Mark Antony", "Ancient"), ("The Code of Hammurabi", "Ancient"),
    ("Ancient Egyptian mummification", "Ancient"), ("Spartan military training", "Ancient"),
    ("The Terracotta Army in China", "Ancient"), ("Early Christianity in Rome", "Ancient"),
    ("Pompeii and Mount Vesuvius", "Ancient"), ("The Oracle of Delphi", "Ancient"),
    ("Mesopotamian irrigation systems", "Ancient"), ("The Peloponnesian War", "Ancient"),
    ("Ancient Mayan calendar", "Ancient"), ("Stonehenge and its origins", "Ancient"),
    ("The Olympic Games in Greece", "Ancient"), ("Confucius and his teachings", "Ancient"),
    ("The Great Wall of China construction", "Ancient"), ("The Library of Alexandria", "Ancient"),
    ("The Han Dynasty's rise", "Ancient"), ("Ancient Persian Empire", "Ancient"),
    ("The Trojan War legends", "Ancient"), ("Mesopotamian Ziggurats", "Ancient"),
    
    #  Medieval History 
    ("What was the Magna Carta?", "Medieval"), ("The Hundred Years War", "Medieval"),
    ("Who was Joan of Arc?", "Medieval"), ("The Black Death pandemic", "Medieval"),
    ("Feudalism and the role of knights", "Medieval"), ("The Crusades in the Holy Land", "Medieval"),
    ("The signing of the Magna Carta", "Medieval"), ("Genghis Khan and the Mongols", "Medieval"),
    ("The Byzantine Empire's peak", "Medieval"), ("Charlemagne and the Holy Roman Empire", "Medieval"),
    ("The Vikings' raids on England", "Medieval"), ("Samurai in Feudal Japan", "Medieval"),
    ("The Battle of Hastings (1066)", "Medieval"), ("The rise of the Ottoman Empire", "Medieval"),
    ("Medieval castle architecture", "Medieval"), ("The Spanish Reconquista", "Medieval"),
    ("Serfdom and peasant life", "Medieval"), ("The Silk Road trade routes", "Medieval"),
    ("The Plague of Justinian", "Medieval"), ("Knights Templar history", "Medieval"),
    ("Gothic cathedrals and stained glass", "Medieval"), ("The Islamic Golden Age", "Medieval"),
    ("Macroeconomics in medieval cities", "Medieval"), ("The War of the Roses", "Medieval"),
    ("The arrival of the Moors in Spain", "Medieval"), ("Byzantine mosaics and art", "Medieval"),
    ("The Council of Clermont", "Medieval"), ("Marco Polo's travels", "Medieval"),

    # Modern History
    ("Why did World War 1 start?", "Modern"), ("Industrial Revolution causes", "Modern"),
    ("Cold War tensions USA and USSR", "Modern"), ("The fall of the Berlin Wall", "Modern"),
    ("Adolf Hitler and World War 2", "Modern"), ("Atomic bombing of Hiroshima", "Modern"),
    ("Space Race and moon landing", "Modern"), ("The French Revolution and Napoleon", "Modern"),
    ("American Civil War and Lincoln", "Modern"), ("The Russian Revolution 1917", "Modern"),
    ("Nelson Mandela and Apartheid", "Modern"), ("The Vietnam War and protests", "Modern"),
    ("The invention of the steam engine", "Modern"), ("Decolonization of Africa", "Modern"),
    ("The rise of the Internet", "Modern"), ("The Great Depression 1929", "Modern"),
    ("Martin Luther King Jr. speech", "Modern"), ("The Cuban Missile Crisis", "Modern"),
    ("The Suffragette movement", "Modern"), ("The Meiji Restoration Japan", "Modern"),
    ("Gorbachev and Perestroika", "Modern"), ("The Indian Independence movement", "Modern"),
    ("Mahatma Gandhi's non-violence", "Modern"), ("The Invention of Radar", "Modern"),
    ("Space Shuttle Challenger disaster", "Modern"), ("The 9/11 attacks impact", "Modern"),
    ("Digital Revolution and computing", "Modern"), ("The formation of the EU", "Modern"),
]

def train():
    print("🚀 Starting Historical Classifier Training...")
    
  
    df = pd.DataFrame(DATA, columns=["text", "era"])
    

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC(C=1.0))
    ])
    
    model.fit(df['text'], df['era'])
    
    # Save the Model (As a .joblib file)
    joblib.dump(model, 'history_classifier.joblib')
    print("✅ DONE: Trained model saved as 'history_classifier.joblib'!")

if __name__ == "__main__":
    train()
