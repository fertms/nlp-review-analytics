# Coleta e geração de dados de reviews para análise NLP
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
OUTPUT = Path("../data/raw")
OUTPUT.mkdir(exist_ok=True)

# Reviews simuladas realistas por categoria
reviews_data = {
    "Electronics": {
        "positive": [
            "Amazing product! Works perfectly and the battery life is outstanding.",
            "Excellent quality, fast delivery. Exactly as described. Highly recommend!",
            "Great value for money. Setup was easy and performance is impressive.",
            "Best purchase I made this year. Works flawlessly, very happy with it.",
            "Fantastic build quality. Exceeded my expectations in every way.",
            "Love this product! Fast, reliable and looks great. Worth every penny.",
            "Top notch quality. Customer service was also very helpful.",
            "Incredible performance. Way better than my previous device.",
            "Perfect product, arrived quickly and works exactly as advertised.",
            "Outstanding quality and design. Would definitely buy again.",
        ],
        "neutral": [
            "Decent product for the price. Does what it's supposed to do.",
            "It's okay. Nothing special but gets the job done.",
            "Average quality. Expected more features for this price range.",
            "Works fine. Delivery took longer than expected though.",
            "Product is acceptable. Instructions could be clearer.",
            "Not bad, not great. Does the basic functions well enough.",
            "Reasonable quality. A few minor issues but overall acceptable.",
            "Standard product. Works as described, nothing extraordinary.",
        ],
        "negative": [
            "Very disappointed. Product stopped working after two weeks.",
            "Poor quality. Broke easily and customer service was unhelpful.",
            "Terrible experience. Product arrived damaged and return process is awful.",
            "Not as described. Cheap materials and weak performance.",
            "Waste of money. Completely stopped functioning after first use.",
            "Horrible product. Would not recommend to anyone.",
            "Very bad quality. Already showing signs of wear after 3 days.",
            "Disappointed with purchase. Looks nothing like the pictures.",
        ]
    },
    "Books": {
        "positive": [
            "Absolutely captivating! Couldn't put it down. A masterpiece.",
            "Brilliant writing. One of the best books I've read this decade.",
            "Fascinating story with deep characters. Highly recommended.",
            "Excellent read! Thought-provoking and beautifully written.",
            "Loved every page. The author's storytelling is exceptional.",
            "Incredible book. Changed my perspective on many things.",
            "Wonderful narrative. Engaging from the first to the last page.",
            "Must read! Insightful, entertaining and very well written.",
            "Outstanding book. The plot twists kept me guessing throughout.",
            "Perfect blend of education and entertainment. Truly remarkable.",
        ],
        "neutral": [
            "Decent book. Interesting premise but execution was average.",
            "Okay read. Some good parts but overall felt incomplete.",
            "Average writing. The story had potential but fell short.",
            "Not bad. A few interesting chapters but mostly predictable.",
            "Readable but forgettable. Nothing that stands out.",
            "Mediocre at best. Expected more depth from this author.",
        ],
        "negative": [
            "Extremely boring. Could not finish it.",
            "Very disappointing. The story goes nowhere.",
            "Terrible writing. Full of plot holes and inconsistencies.",
            "Waste of time. Nothing happens for 200 pages.",
            "Dreadful book. The characters are flat and uninteresting.",
            "Not worth reading. Poorly written and badly edited.",
        ]
    },
    "Clothing": {
        "positive": [
            "Perfect fit! Great quality fabric and beautiful design.",
            "Love this item! Exactly as pictured and very comfortable.",
            "Excellent quality. Well made and the color is gorgeous.",
            "Great purchase! Comfortable, stylish and true to size.",
            "Amazing quality for the price. Very satisfied with this buy.",
            "Beautiful piece. The material feels premium and fits perfectly.",
            "Fantastic clothing item. Got many compliments wearing it.",
            "Very happy with this purchase. Quality exceeded expectations.",
        ],
        "neutral": [
            "Okay quality. Fits as expected but nothing remarkable.",
            "Decent item. Color slightly different from pictures.",
            "Average quality. Fine for casual use.",
            "Not bad. Took a while to arrive but product is acceptable.",
            "Reasonable purchase. Material is okay but not outstanding.",
        ],
        "negative": [
            "Very poor quality. Stitching came apart after first wash.",
            "Terrible fit. Size chart is completely wrong.",
            "Disappointed. Color faded immediately after washing.",
            "Poor material. Feels cheap and uncomfortable.",
            "Not as described. Much lower quality than pictures suggest.",
            "Awful product. Returned immediately.",
        ]
    },
    "Food & Kitchen": {
        "positive": [
            "Excellent product! Works better than expected in the kitchen.",
            "Amazing quality. Makes cooking so much easier and enjoyable.",
            "Love this kitchen tool! Durable, efficient and easy to clean.",
            "Perfect for everyday use. Well designed and very practical.",
            "Great product! Family loves it. Worth every cent.",
            "Outstanding kitchen item. Professional quality at home price.",
            "Brilliant purchase. Changed my cooking experience completely.",
        ],
        "neutral": [
            "Works fine. Nothing special but does the job.",
            "Acceptable quality. A bit tricky to clean though.",
            "Decent product. Instructions were confusing at first.",
            "Average kitchen item. Does what it promises, nothing more.",
        ],
        "negative": [
            "Very disappointed. Broke after minimal use.",
            "Terrible quality. Not dishwasher safe despite claims.",
            "Poor design. Difficult to use and even harder to clean.",
            "Waste of money. Stopped working within a month.",
            "Not as advertised. Much smaller than shown in pictures.",
        ]
    }
}

# Gerar dataset
records = []
review_id = 1
categories = list(reviews_data.keys())
sentiments = ["positive", "neutral", "negative"]
ratings_map = {"positive": [4, 5], "neutral": [3], "negative": [1, 2]}

for category, sentiment_reviews in reviews_data.items():
    for sentiment, texts in sentiment_reviews.items():
        for text in texts:
            # Adiciona variações
            for _ in range(np.random.randint(2, 5)):
                rating = np.random.choice(ratings_map[sentiment])
                helpful_votes = np.random.randint(0, 50) if sentiment == "positive" else np.random.randint(0, 20)
                records.append({
                    "review_id"    : f"R{str(review_id).zfill(5)}",
                    "category"     : category,
                    "rating"       : rating,
                    "review_text"  : text,
                    "sentiment_label": sentiment,
                    "helpful_votes": helpful_votes,
                    "verified_purchase": np.random.choice([True, False], p=[0.8, 0.2]),
                    "review_date"  : pd.Timestamp("2023-01-01") + pd.Timedelta(days=np.random.randint(0, 365)),
                })
                review_id += 1

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT / "reviews.csv", index=False, encoding="utf-8-sig")

print(f"Dataset gerado: {len(df)} reviews")
print(f"Categorias: {df['category'].value_counts().to_dict()}")
print(f"Sentimentos: {df['sentiment_label'].value_counts().to_dict()}")
print(f"Rating médio: {df['rating'].mean():.2f}")