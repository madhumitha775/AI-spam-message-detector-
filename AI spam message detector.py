from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

messages = [
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free gift today",
    "Click here to win cash",
    "Get a free reward now",
    "Meeting is scheduled for tomorrow",
    "Please send me your notes",
    "Are you coming to college",
    "Your assignment is completed",
    "Let us meet this evening",
    "Please call me when you are free",
    "The class starts at nine"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Normal", "Normal", "Normal", "Normal",
    "Normal", "Normal", "Normal"
]

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(messages, labels)

print("AI SPAM MESSAGE DETECTOR")
print("Type 'exit' to stop.")

while True:
    message = input("\nEnter a message: ")

    if message.lower() == "exit":
        print("Program ended.")
        break

    prediction = model.predict([message])[0]
    print("Prediction:", prediction)
