from atproto import Client

# log in blusky api
client = Client()
handle = 'jaschparmar.bsky.social'
app_password = '4jj5-4zo6-m4ez-t3i2'
client.login(handle, app_password)

trial = ['jay.bsky.team']

#for target in target_handles_BOT:
for target in trial:
    try:
        profile = client.app.bsky.actor.get_profile({'actor': target})

        # Fetch recent posts (up to 100)
        try:
            response = client.app.bsky.feed.get_author_feed({'actor': target, 'limit': 100})
            posts = [post for post in response['feed'] ]
            print(f"🔍 Retrieved {len(posts)} posts for {target}")
        except Exception:
            posts = []

    except Exception as e:
        print(f"❌ Failed to fetch {target}: {e}")

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \
"" \
"" \
"" \
"")
for post_view in posts:
    print(post_view['post']['like_count'])