from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

reviews = [
    "The new smartphone has an amazing camera and battery life!",
    "The delivery was late and the packaging was damaged."
]

for review in reviews:
    result = sentiment_analyzer(review)[0]
    print(
        f"Review: {review}\n"
        f" -> {result['label']} ({round(result['score'], 3)})\n"
    )

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

document = "The central bank raised interest rates to control rising inflation."
candidate_labels = ["Politics", "Economy", "Sports", "Technology"]

classification = classifier(document, candidate_labels)

print("Document:", document)
for label, score in zip(
    classification["labels"], classification["scores"]
):
    print(f"{label}: {round(score, 3)}")
