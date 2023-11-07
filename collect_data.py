import praw
from praw.models import MoreComments
import pandas as pd
import numpy as np
import os


# Gets the top post IDs
def get_post_ids(subreddit_name, post_limit=None):
    posts = reddit.subreddit(subreddit_name).top(limit=post_limit)
    post_ids = [post.id for post in posts]
    return post_ids


def get_top_two_level_comments(submission, replace_more_limit=None):
    comments = []
    submission.comments.replace_more(limit=replace_more_limit)
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        comments.append(top_level_comment.body)
        for second_level_comment in top_level_comment.replies:
            if isinstance(second_level_comment, MoreComments):
                continue
            comments.append(second_level_comment.body)
    return comments

# mines posts and comments from specified list of subreddits
def gather_data(subreddit_list, n):
    # grab post ids in the specified subreddits
    post_ids =[]
    for subreddit_name in subreddit_list:
        post_ids.extend(get_post_ids(subreddit_name, post_limit=n))
    print("got post ids")

    # grab titles of the post-ids
    post_titles = []
    for id in post_ids:
        submission = reddit.submission(id=id)
        post_titles.append(submission.title)
    print("got titles")

    # grab comments for the post-ids
    all_comments = []
    for id in post_ids:
        submission = reddit.submission(id)

        # Get top two comment levels for posts.
        post_comments = get_top_two_level_comments(submission, replace_more_limit=5)
        all_comments.append(post_comments)
    print("got comments")

    # Save the data to a pandas dataframe. The comments are a list of comments for each id/title
    df = pd.DataFrame({'ids': post_ids, 'titles': post_titles, 'comments': all_comments})
    header = "ids|*|titles|*|comments"
    # df.to_csv('reddit_data', sep='|') # "|" may not be the best sep token, since maybe someone types that? you may want to use something more complex (e.g. |*|)
    np.savetxt(os.path.join('data', 'reddit_data_neg'), df, fmt=["%s", '%s', '%s'], delimiter="|*|", header=header, encoding='utf-8', comments='')


if __name__ == "__main__":
    # instantiate an instance of the Reddit API
    reddit = praw.Reddit(client_id='QXyXx6ngHJhTFP7gLBs29w',
                         client_secret='0-z5nYLk2e7FWzmSi1yExAd9DiYCTA',
                         user_agent='Data Miner')
    print("API instance initialized")

    subreddit_list = ['electronic_cigarette', 'VapeWild', 'QuitVaping', 'Vape_Chat', 'juul', 'E_Cigarette']
    # subreddit_list = ['electronic_cigarette', 'Minecraft']
    # subreddit_list = ['QuitVaping']
    number_of_posts_from_each = 1000
    gather_data(subreddit_list, number_of_posts_from_each)
