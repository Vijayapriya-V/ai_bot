# Import libraries
import pandas as pd
import spacy
import requests
import io

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your ball-by-ball dataset from Google Drive
file_id = "1PeOj9sb_sH65KEzfwLv--GKsB81uK01w"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(download_url)
df = pd.read_csv(io.StringIO(response.text))
print("Dataset Loaded! Shape:", df.shape)

# Memory for learned answers
learned_answers = {}

# Chatbot logic
def process_query(query):
    doc = nlp(query.lower())
    tokens = [token.lemma_ for token in doc]

    # Check learned answers first
    for learned_q in learned_answers:
        if learned_q in query.lower():
            return learned_answers[learned_q]

    # 1. Columns
    if "column" in tokens or "feature" in tokens:
        return f"Columns in dataset: {', '.join(df.columns)}"

    # 2. Total balls
    elif "ball" in tokens and "count" in tokens:
        return f"Total number of balls: {len(df)}"

    # 3. Top scorer (by player ID)
    elif "top" in tokens and "scorer" in tokens:
        scored_df = df[df["Batsman_Scored"].apply(lambda x: str(x).isdigit())]
        scored_df["Batsman_Scored"] = scored_df["Batsman_Scored"].astype(int)
        top_batsman = scored_df.groupby("Striker_Id")["Batsman_Scored"].sum().idxmax()
        runs = scored_df.groupby("Striker_Id")["Batsman_Scored"].sum().max()
        return f"Top scorer (by ID): Player {top_batsman} with {runs} runs"

    # 4. First rows
    elif "top 5" in query or "head" in query:
        return df.head().to_string(index=False)

    # 5. Match count
    elif "match" in tokens and "count" in tokens:
        return f"Total number of matches: {df['Match_Id'].nunique()}"

    # If unknown
    else:
        return None  # Indicates unknown answer

# Chat loop with learning
print("\nWelcome to the AI-Powered IPL Chatbot!")
while True:
    user_input = input("\nAsk a question (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    response = process_query(user_input)

    if response:
        print("Bot:", response)
    else:
        print("Bot: I'm not sure how to answer that. Can you please tell me the correct answer?")
        user_answer = input("You: ")
        learned_answers[user_input.lower()] = user_answer
        print("Bot: Got it! I'll remember that for next time.")