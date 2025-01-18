import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  


reddit = praw.Reddit(
    client_id="CYqXfUvPEcSyxR0DUOTSkQ",
    client_secret="2rzZttEAcvOV3SCEk-dYXB0QEtujjA",
    user_agent="Gaming Chat Scraper"
)


subreddit_names = ["leagueoflegends", "FortNiteBR", "gaming", "gamingcirclejerk"]
comments = []


keywords = [
    "noob", "trash", "idiot", "loser", "stupid", "ban", "kill yourself",
    "quit", "report", "hacker", "scam", "fool", "hate", "toxic",
    "garbage", "worthless", "cheater", "retard", "crybaby"
]


analyzer = SentimentIntensityAnalyzer()


def is_toxic(comment):
    sentiment = analyzer.polarity_scores(comment)
    return sentiment['compound'] < -0.4  


for name in subreddit_names:
    try:
        print(f"Scraping comments from r/{name}...")
        subreddit = reddit.subreddit(name)
        
        
        for submission in subreddit.controversial(limit=10):  
            submission.comments.replace_more(limit=0)  
            
            for comment in submission.comments.list():
                
                if any(word in comment.body.lower() for word in keywords) or is_toxic(comment.body):
                    if len(comment.body) > 50:  
                        comments.append(comment.body)

    except Exception as e:
        print(f"Error accessing r/{name}: {e}")


df = pd.DataFrame(comments, columns=["comment_text"])
df.to_csv("purely_toxic_comments.csv", index=False)

print(f"Scraped and filtered {len(comments)} toxic comments. Saved to 'purely_toxic_comments.csv'.")
