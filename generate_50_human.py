from atproto import Client
import json
import random

client = Client()
handle = 'jaschparmar.bsky.social'
app_password = '4jj5-4zo6-m4ez-t3i2'
client.login(handle, app_password)

client = Client()
client.login(handle, app_password)

# Get follows of jay.bsky.team
follows_response = client.app.bsky.graph.get_follows({'actor': 'jay.bsky.team'})
follows = follows_response['follows']

# Shuffle and sample 50 follows
random.shuffle(follows)
sample_size = min(50, len(follows))
sampled_follows = follows[:sample_size]

# Extract just the handles
human_handles = [user['handle'] for user in sampled_follows]

# Save to JSON
with open('50_human.json', 'w', encoding='utf-8') as f:
    json.dump(human_handles, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {sample_size} human handles to human_handles.json")
