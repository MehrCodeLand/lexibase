import numpy as np
from sklearn.cluster import MiniBatchKMeans
from core.embeddings import get_vector
from core.qdrant_client import client, COLLECTION_NAME
from tqdm import tqdm
import json
import requests
import os
from typing import List, Dict
from loguru import logger


AVALAI_API_KEY = os.getenv("AVALAI_API_KEY", "your-api-key-here")
AVALAI_API_URL = "https://api.avalai.ir/v1/chat/completions"


# Fetches all words and their vector embeddings from Qdrant database
# Returns two lists: words as strings and their corresponding vector arrays
# Uses scrolling to handle large datasets efficiently with batching
def fetch_all_words_from_qdrant(batch_size=1000):
    all_words = []
    all_vectors = []
    
    offset = None
    
    print("Fetching words from Qdrant...")
    while True:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True
        )
        
        points, next_offset = results
        
        if not points:
            break
            
        for point in points:
            all_words.append(point.payload['word'])
            all_vectors.append(point.vector)
        
        offset = next_offset
        if next_offset is None:
            break
    
    print(f"Total words fetched: {len(all_words)}")
    return all_words, np.array(all_vectors)


# Performs K-means clustering on word vectors to group semantically similar words
# Uses MiniBatchKMeans for memory efficiency with large datasets
# Returns clusters dict with words sorted by distance to cluster center
def cluster_all_words(words, vectors, num_clusters=50):
    print(f"Clustering {len(words)} words into {num_clusters} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=42,
        batch_size=1000,
        max_iter=100,
        verbose=1
    )
    
    labels = kmeans.fit_predict(vectors)
    
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'word': words[idx],
            'distance': float(np.linalg.norm(vectors[idx] - kmeans.cluster_centers_[label]))
        })
    
    for cluster_id in clusters:
        clusters[cluster_id] = sorted(clusters[cluster_id], key=lambda x: x['distance'])
    
    print(f"Clustering completed! Created {len(clusters)} clusters")
    return clusters, kmeans


# Calls AvalAI LLM API to generate a 3-word semantic title for a word cluster
# Sends top 45 representative words and uses prompt engineering to get concise titles
# Includes retry logic and fallback naming if API fails
def generate_cluster_title_with_avalai(representative_words: List[str], max_retries=3) -> str:
    words_string = ", ".join(representative_words[:45])
    
    prompt = f"""You are an expert linguist analyzing word clusters. 
Given the following related words: {words_string}

Generate EXACTLY 3 words (separated by underscores) that best describe the common theme or category of these words.
The title should be concise, descriptive, and capture the semantic essence.

Examples:
- If words are: doctor, nurse, hospital, medicine → Medical_Health_Care
- If words are: computer, software, programming, code → Technology_Computing_Software
- If words are: happy, joy, excited, cheerful → Positive_Emotions_Feelings

Respond with ONLY the 3-word title, nothing else."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AVALAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates concise category titles."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 20
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                AVALAI_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                title = result['choices'][0]['message']['content'].strip()
                
                title = title.replace(" ", "_")
                title = ''.join(c for c in title if c.isalnum() or c == '_')
                
                return title
            else:
                print(f"AvalAI API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error calling AvalAI (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in 2 seconds...")
            import time
            time.sleep(2)
    
    return f"Cluster_{representative_words[0]}_{representative_words[1]}_{representative_words[2]}"


# Processes all clusters to generate semantic titles using AvalAI LLM
# Takes top N representative words from each cluster and calls LLM for naming
# Returns structured dictionary with titles, metadata, and word lists
def assign_category_names_with_llm(clusters: Dict, top_n=45) -> Dict:
    categorized_clusters = {}
    
    print(f"\nGenerating semantic titles for {len(clusters)} clusters using AvalAI...")
    
    for cluster_id, words_data in tqdm(clusters.items(), desc="Naming clusters with LLM"):
        top_words = [w['word'] for w in words_data[:top_n]]
        
        category_title = generate_cluster_title_with_avalai(top_words)
        
        categorized_clusters[category_title] = {
            'cluster_id': int(cluster_id),
            'title': category_title,
            'representative_words': top_words[:15],
            'total_words': len(words_data),
            'sample_words': [w['word'] for w in words_data[:30]],
            'all_words': [w['word'] for w in words_data]
        }
        
        print(f"  Cluster {cluster_id} → {category_title}")
    
    return categorized_clusters


# Saves the categorized clusters dictionary to a JSON file
# Uses UTF-8 encoding to handle special characters properly
def save_clusters(clusters: Dict, filename='word_categories.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Clusters saved to {filename}")


# Prints a formatted summary of the categorization results
# Shows total statistics and sample categories with example words
def print_summary(categorized: Dict):
    print("\n" + "="*80)
    print("📊 WORD CATEGORIZATION SUMMARY")
    print("="*80)
    
    print(f"\nTotal Categories: {len(categorized)}")
    print(f"Total Words Categorized: {sum(data['total_words'] for data in categorized.values())}")
    
    print("\n🏷️  Sample Categories:\n")
    
    for i, (category_name, data) in enumerate(list(categorized.items())[:10]):
        print(f"{i+1}. {category_name}")
        print(f"   Words: {data['total_words']}")
        print(f"   Examples: {', '.join(data['representative_words'][:8])}")
        print()


# Main orchestration function that executes the complete clustering pipeline
# Steps: fetch words → cluster → generate titles with LLM → save results
# Also saves the K-means model for future predictions
def main():
    words, vectors = fetch_all_words_from_qdrant()
    
    if len(words) == 0:
        print("❌ No words found in Qdrant. Please run load_nltk_words.py first.")
        return
    
    num_clusters = 50
    clusters, kmeans_model = cluster_all_words(words, vectors, num_clusters)
    
    categorized = assign_category_names_with_llm(clusters, top_n=45)
    
    save_clusters(categorized, 'word_categories.json')
    
    print_summary(categorized)
    
    # NEW: Update Qdrant with categories
    from core.qdrant_client import update_words_with_categories
    print("\n🔄 Updating Qdrant database with categories...")
    stats = update_words_with_categories(categorized)
    print(f"✅ Updated {stats['total_updated']} words in Qdrant")
    print(f"❌ Failed: {stats['total_failed']} words")
    
    import pickle
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print("\n✅ K-means model saved to kmeans_model.pkl")
    
    return categorized, kmeans_model


if __name__ == "__main__":
    if AVALAI_API_KEY == "your-api-key-here":
        print("⚠️  Warning: Please set AVALAI_API_KEY environment variable")
        print("   export AVALAI_API_KEY='your-actual-api-key'")
    
    categories, model = main()