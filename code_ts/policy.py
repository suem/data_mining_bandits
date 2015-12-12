
import numpy.random

# Prior parameters of a Beta distribution.
ALPHA = 1
BETA = 1

# Dictionaries for counting successes and failures.
S = {}
F = {}

previous_recommendation = None


def set_articles(articles):
    pass


def update(reward):
    if reward == 1:
        S[previous_recommendation] += 1
    elif reward == 0:
        F[previous_recommendation] += 1


def recommend(time, user_features, article_ids):
    global previous_recommendation
    largest_sample = -1.0
    best_article = article_ids[0]
    for article in article_ids:
        if not S.has_key(article):
            S[article] = 0
            F[article] = 0
        sample = numpy.random.beta(ALPHA + S[article], BETA + F[article])
        if sample > largest_sample:
            largest_sample = sample
            best_article = article
    previous_recommendation = best_article
    return best_article


def reccomend(time, user_features, articles):
    return recommend(time, user_features, articles)

