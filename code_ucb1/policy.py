import numpy as np
import numpy.random

ALPHA = 1.0

# dictionary to keep all M matrices for each article x
u_x = {}
n_x = {}

last_id = -1

t = 1


def set_articles(articles):
    pass


def update(reward):
    global t
    t += 1

    y = reward
    if not y == -1:
        article_id = last_id
        n = n_x[article_id] + 1
        n_x[article_id] = n
        u = u_x[article_id]
        u_x[article_id] = u + (y-u)/n


def recommend(time, user_features, article_ids):
    global last_id

    id_max = np.random.choice(article_ids)
    ucb_max = -1
    for article_id in article_ids:
        if not u_x.has_key(article_id):
            u_x[article_id] = 0
            n_x[article_id] = 0
        u = u_x[article_id]
        n = n_x[article_id]
        if n == 0:
            # this arm was never played before, play it
            last_id = article_id
            return article_id

        ucb = u + ALPHA * np.sqrt(2 * np.log(t) / n)
        if ucb > ucb_max:
            ucb_max = ucb
            id_max = article_id

    last_id = id_max
    return id_max


# this is the version that will be called. Leave it as is
def reccomend(time, user_features, articles):
    return recommend(time, user_features, articles)

