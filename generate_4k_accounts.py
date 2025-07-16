import random
import json
from atproto import Client

# === Config ===
handle = 'jaschparmar.bsky.social'
app_password = '4jj5-4zo6-m4ez-t3i2'
official_account = 'bsky.app'
followers_to_fetch = 20000  # We'll stop early if we reach 4321 active users
min_posts_required = 5
target_sample_size = 4321

# === Connect to Bluesky ===
client = Client()
client.login(handle, app_password)

# === Fetch + Filter Followers ===
valid_handles = []
seen_handles = set()
cursor = None

print(f"🔄 Collecting followers of {official_account}...")

while len(valid_handles) < target_sample_size:
    response = client.app.bsky.graph.get_followers({
        'actor': official_account,
        'limit': 100,
        'cursor': cursor
    })

    followers_batch = response['followers']
    cursor = response['cursor']

    for follower in followers_batch:
        handle = follower['handle']

        if handle in seen_handles:
            continue
        seen_handles.add(handle)

        try:
            profile = client.app.bsky.actor.get_profile({'actor': handle})
            if profile['posts_count'] >= min_posts_required:
                valid_handles.append(handle)
                print(f"✅ {handle} — {profile['posts_count']} posts ({len(valid_handles)}/{target_sample_size})")
        except Exception as e:
            print(f"❌ Skipped {handle}: {e}")

        if len(valid_handles) >= target_sample_size:
            break

    if not cursor:
        print("⚠️ Ran out of followers before hitting target.")
        break

print(f"🎯 Final sample: {len(valid_handles)} valid accounts with ≥ {min_posts_required} posts")

# === Save to file ===
with open('4k_accounts.json', 'w') as f:
    json.dump(valid_handles, f, indent=2)

print("✅ Saved to 4k_accounts.json")