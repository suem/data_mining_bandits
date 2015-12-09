import numpy as np
import numpy.random


# number of dimensions (user features and article features, they are the same here
N = 6
ALPHA = 1

# dictionary to keep all M matrices for each article x
M_x = {}
b_x = {}
Articles = {}

last_id = np.array([])
last_z = np.array([])


def set_articles(articles):
    for a in articles:
        article_id = a[0]
        Articles[article_id] = a[1:]
        M_x[article_id] = np.eye(N)
        b_x[article_id] = np.zeros(N)


def update(reward):
    y = reward
    article_id = last_id
    if Articles.has_key(article_id):
        z = last_z
        M_x[article_id] += z.dot(z)
        b_x[article_id] += y*z


def recommend(time, user_features, article_ids):
    global last_z
    global last_id
    ucb_max = -1
    x_max = np.zeros(N)
    id_max = np.random.choice(article_ids, size=1)[0]
    z = np.array(user_features)

    for i in range(len(article_ids)):
        article_id = article_ids[i]
        if Articles.has_key(article_id):
            x = Articles[article_id]
            m = M_x[article_id]
            b = b_x[article_id]
            w = np.linalg.solve(m, b)
            ucb = w.dot(z) + ALPHA*np.sqrt(z.dot(np.linalg.solve(m, z)))
            if ucb_max < ucb:
                ucb_max = ucb
                id_max = article_id
    last_z = z
    last_id = id_max
    return id_max


# this is the version that will be called. Leave it as is
def reccomend(time, user_features, articles):
    return recommend(time, user_features, articles)

