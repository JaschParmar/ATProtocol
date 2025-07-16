
from atproto import Client
import json
from datetime import datetime
import re
from collections import Counter
from datetime import datetime, timezone


# log in blusky api
client = Client()
handle = 'jaschparmar.bsky.social'
app_password = '4jj5-4zo6-m4ez-t3i2'
client.login(handle, app_password)

jay = ['jay.bsky.team']

# list of bot_profiles
target_handles_BOT = [
    'robertswithcodes.bsky.social',
    'apo11o.bsky.social',
    'rosecoopermz.bsky.social',
    'mrmasa.bsky.social',
    'howtoguide.bsky.social',
    'lucy-my.bsky.social',
    'amzndaily.bsky.social',
    'thepriapicknave.bsky.social',
    'jettrack.bsky.social',
    'agshdjdjj.bsky.social',
    'dallas-alerts.bsky.social',
    'vcbjeiyeeg7.bsky.social',
    'ai.bots.law',
    'tibrnowplayingbot.mastodon.social.ap.brid.gy',
    'deans-girl.bsky.social',
    'ysbsbsbsn.bsky.social',
    'dealradarx.com',
    'newsflash.one',
    'mistersalesman.bsky.social',
    'thefactsdaily.bsky.social',
    'xelansignal.bsky.social',
    'breedable.tech',
    'baconpress.bsky.social',
    'kikisnags.bsky.social',
    'tvgrradio.co.uk',
    'toulousetrafic.com',
    'thenewstribune.com',
    'theolympian.com',
    'tri-cityherald.com',
    'idahostatesman.com',
    'joshlanyon.bsky.social',
    'winetopics.bsky.social',
    'zonecrypto.bsky.social',
    'investorfazal.bsky.social',
    'ahmetbutun.bsky.social',
    'boostlensats.bsky.social',
    'speedrun-new.bsky.social',
    'countingsheep.bsky.social',
    'justabeardedgay.com',
    'radiojammor.bsky.social',
    'central-dyk.bsky.social',
    'stupidgirltwink.bsky.social',
    'defendinfo2025.bsky.social',
    'misabot.bsky.social',
    'longtail.news',
    'acarsdrama.bsky.social',
    'apveng.bsky.social',
    'europesays.bsky.social',
    'flakefood.bsky.social',
    'arduinolibs.bsky.social',
    'tubeoftheday.bsky.social',
    'neogotmystats.bsky.social',
    'tsvndra.bsky.social',
    'nsfwtiktok.bsky.social',
    'bigboobsgw.bsky.social',
]

# list of human_profiles
target_handles_HUMAN = [
    'jay.bsky.team',
    'jaschparmar.bsky.social',
    'pfrazee.com',
    "samir.fedica.com",
    "usafacts.org",
    "yimbyland.com",
    "mada299.bsky.social",
    "wisdompedlars.bsky.social",
    "kenjennings.bsky.social",
    "adobe.com",
    "cyberplantae.bsky.social",
    "billgates.bsky.social",
    "lexfridman.bsky.social",
    "unusualwhales.bsky.social",
    "mrjamesob.bsky.social",
    "gestaltu.bsky.social",
    "fosse.co",
    "calabro.io",
    "d3ol.dev",
    "utopia-defer.red",
    "alextamkin.bsky.social",
    "dougorgan.bsky.social",
    "senatorshoshana.bsky.social",
    "sleepinyourhat.bsky.social",
    "kamalaharris.com",
    "deanwb.bsky.social",
    "somewhere.systems",
    "metr.org",
    "weelplanner.bsky.social",
    "larmar5.bsky.social",
    "hunterw.bsky.social",
    "blockpartyapp.com",
    "ericries.bsky.social",
    "jennycohn.bsky.social",
    "robbysoave.bsky.social",
    "jeopardyofficial.bsky.social",
    "interfluidity.com",
    "skylight.social",
    "mattyglesias.bsky.social",
    "toronto.ca",
    "lifewinning.com",
    "lculbs.bsky.social",
    "sculptedreef.com",
    "rickklau.com",
    "ashleyrgold.bsky.social",
    "nico-encounter.bsky.social",
    "deamplified.com",
    "darrin.bsky.team",
    "alankadima.bsky.social",
    "michaelgarfield.bsky.social",
    "maiab.bsky.social",
    "colinc.bsky.social",
    "lionessgames.bsky.social"

]

# collect dataset
dataset = []

# data collector
for target in target_handles_HUMAN + target_handles_BOT:
    try:
        profile = client.app.bsky.actor.get_profile({'actor': target})

        # Fetch recent posts (up to 100)
        try:
            response = client.app.bsky.feed.get_author_feed({'actor': target, 'limit': 100})
            posts = [post for post in response['feed'] ]
            print(f"🔍 Retrieved {len(posts)} posts for {target}")
        except Exception:
            posts = []

        # Compute post behaviour metrics
        timestamps = []
        for post in posts:
            try:
                ts = post['post']['indexed_at']
                ts_obj = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(ts_obj)
            except Exception:
                continue  # skip any post with malformed timestamp

        timestamps.sort(reverse=True)
        intervals = [
            (timestamps[i] - timestamps[i + 1]).total_seconds() / 60
            for i in range(len(timestamps) - 1)
        ]
        average_post_interval = sum(intervals) / len(intervals) if intervals else None


        # --- Post Texts ---
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

        # --- Contains Links Ratio ---
        def has_link(text):
            return bool(re.search(r'http[s]?://', text))

        contains_links_ratio = (
            sum(1 for post_view in posts if has_link(getattr(post_view.post.record, 'text', '')))
            / len(posts) if posts else 0
        )

        # --- Hashtags Per Post ---
        def count_hashtags(text):
            return text.count('#')

        hashtags_per_post = (
            sum(count_hashtags(getattr(post_view.post.record, 'text', '')) for post_view in posts)
            / len(posts) if posts else 0
        )

        # --- Burst Posting ---
        burst = (
            any(
                (timestamps[i] - timestamps[i + 2]).total_seconds() < 300
                for i in range(len(timestamps) - 2)
            ) if len(timestamps) >= 3 else False
        )

        # --- Replies to Others ---
        replies_to_others = sum(
            1 for post_view in posts
            if hasattr(post_view.post.record, 'reply') and post_view.post.record.reply is not None
        )

       # --- Reposts of Others ---
        reposts_of_others = sum(
            1 for post_view in posts
            if post_view.reason is not None 
        )
    
        # --- Likes and Replies Received ---
        like_counts = []
        reply_counts = []

        for post_view in posts:
            try:
                like_counts.append(post_view['post']['like_count'])
                reply_counts.append(post_view['post']['reply_count'])
            except Exception:
                continue  # just skip if any of those fields are missing

        avg_likes_received = sum(like_counts) / len(like_counts) if like_counts else None
        avg_replies_received = sum(reply_counts) / len(reply_counts) if reply_counts else None

        # --- Follow Ratio ---
        followers_following_ratio = (
            profile.followers_count / profile.follows_count
            if profile.follows_count != 0 else None
        )

        # --- Verified ---
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
        
        account_data = {
            # User baseed features
            'handle': profile['handle'],
            'display_name': profile['display_name'],
            'followers_count': profile['followers_count'],
            'follows_count': profile['follows_count'],
            'followers_following_ratio': followers_following_ratio,
            'posts_count': profile['posts_count'],
            'created_at': profile['created_at'],
            'account_age': account_age_days ,
            'description': profile['description'],
            'description_length': description_length,
            'is_verified': is_verified,

            # Post based features
            'average_post_interval': average_post_interval,
            'repeated_post_ratio': repeated_post_ratio,
            'contains_links_ratio': contains_links_ratio,
            'hashtags_per_post': hashtags_per_post,
            'burst_posting': burst,
            "avg_likes_received": avg_likes_received,
            "avg_replies_received": avg_replies_received,
            "replies_to_others": replies_to_others,
            "reposts_of_others": reposts_of_others,
            'label': 'human' if target in target_handles_HUMAN else 'bot'
        }

        dataset.append(account_data)
        print(f"✅ Fetched {target}")

    except Exception as e:
        print(f"❌ Failed to fetch {target}: {e}")

with open('initital_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("✅ Dataset saved to bluesky_accounts.json")
#print(len(target_handles_BOT+target_handles_HUMAN))
#print(posts[42])
