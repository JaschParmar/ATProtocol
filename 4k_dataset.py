from atproto import Client
import json
from datetime import datetime, timezone
import re
from collections import Counter

# === Login ===
client = Client()
handle = 'jaschparmar.bsky.social'
app_password = '4jj5-4zo6-m4ez-t3i2'
client.login(handle, app_password)

# === Load 4k random accounts ===
with open('4k_accounts.json') as f:
    target_handles = json.load(f)

dataset = []

for target in target_handles:
    try:
        profile = client.app.bsky.actor.get_profile({'actor': target})

        # Fetch recent posts
        try:
            response = client.app.bsky.feed.get_author_feed({'actor': target, 'limit': 100})
            posts = [post for post in response['feed']]
            print(f"🔍 Retrieved {len(posts)} posts for {target}")
        except Exception:
            posts = []

        # === Post Timing ===
        timestamps = []
        for post in posts:
            try:
                ts = post['post']['indexed_at']
                ts_obj = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(ts_obj)
            except Exception:
                continue
        timestamps.sort(reverse=True)
        intervals = [
            (timestamps[i] - timestamps[i + 1]).total_seconds() / 60
            for i in range(len(timestamps) - 1)
        ]
        average_post_interval = sum(intervals) / len(intervals) if intervals else None

        # === Post Texts ===
        contents = []
        for post_view in posts:
            try:
                text = post_view.post.record.text.strip().lower()
                contents.append(text)
            except AttributeError:
                continue

        content_counts = Counter(contents)
        repeated = sum(1 for c in contents if content_counts[c] > 1)
        repeated_post_ratio = repeated / len(contents) if contents else 0

        # === Links & Hashtags ===
        def has_link(text): return bool(re.search(r'http[s]?://', text))
        def count_hashtags(text): return text.count('#')

        contains_links_ratio = (
            sum(1 for p in posts if has_link(getattr(p.post.record, 'text', '')))
            / len(posts) if posts else 0
        )
        hashtags_per_post = (
            sum(count_hashtags(getattr(p.post.record, 'text', '')) for p in posts)
            / len(posts) if posts else 0
        )

        # === Burst Posting ===
        burst = (
            any((timestamps[i] - timestamps[i + 2]).total_seconds() < 300
                for i in range(len(timestamps) - 2)) if len(timestamps) >= 3 else False
        )

        # === Replies ===
        replies_to_others = sum(
            1 for p in posts if hasattr(p.post.record, 'reply') and p.post.record.reply is not None
        )

        # --- Reposts of Others ---
        reposts_of_others = sum(
            1 for post_view in posts
            if post_view.reason is not None 
        )

        # === Likes & Replies Received ===
        like_counts = []
        reply_counts = []
        for p in posts:
            try:
                like_counts.append(p['post']['like_count'])
                reply_counts.append(p['post']['reply_count'])
            except Exception:
                continue

        avg_likes_received = sum(like_counts) / len(like_counts) if like_counts else None
        avg_replies_received = sum(reply_counts) / len(reply_counts) if reply_counts else None

        # === Follower-Following Ratio ===
        followers_following_ratio = (
            profile.followers_count / profile.follows_count
            if profile.follows_count != 0 else None
        )

        # === Verified ===
        is_verified = (
            profile.verification.verified_status == 'valid'
            if hasattr(profile, 'verification') and hasattr(profile.verification, 'verified_status')
            else False
        )

        # --- Account Age ---
        created_at = datetime.fromisoformat(profile['created_at'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        account_age_days = (now - created_at).days

        # --- Description Length ---
        description = profile['description'] or ''
        description_length = len(description.strip())

        # === Save final record ===
        account_data = {
            'handle': profile['handle'],
            'display_name': profile['display_name'],
            'followers_count': profile['followers_count'],
            'follows_count': profile['follows_count'],
            'followers_following_ratio': followers_following_ratio,
            'posts_count': profile['posts_count'],
            'created_at': profile['created_at'],
            'account_age': account_age_days,
            'description': profile['description'],
            'description_length': description_length,
            'is_verified': is_verified,
            'average_post_interval': average_post_interval,
            'repeated_post_ratio': repeated_post_ratio,
            'contains_links_ratio': contains_links_ratio,
            'hashtags_per_post': hashtags_per_post,
            'burst_posting': burst,
            'avg_likes_received': avg_likes_received,
            'avg_replies_received': avg_replies_received,
            'replies_to_others': replies_to_others,
            'reposts_of_others': reposts_of_others
            # No label here — you'll predict that
        }

        dataset.append(account_data)
        print(f"✅ Processed {target}")

    except Exception as e:
        print(f"❌ Failed to process {target}: {e}")

# === Save Dataset ===
with open('unlabeled_4k_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print("✅ Done! Saved to unlabeled_4k_dataset.json")
print(len(dataset))