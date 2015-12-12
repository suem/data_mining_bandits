import numpy as np
import numpy.random

# number of dimensions (user features and article features, they are the same here
N = 6
ALPHA = 0.28

# dictionary to keep all M matrices for each article x
M_x = {}
b_x = {}

last_id = np.array([])
last_z = np.array([])


def set_articles(articles):
    pass
    # for a in articles:
    #     article_id = a[0]
    #     M_x[article_id] = np.eye(N)
    #     b_x[article_id] = np.zeros(N)


def update(reward):
    y = reward
    if not y == -1:
        article_id = last_id
        z = last_z
        m = M_x[article_id]
        mz = m.dot(z)
        M_x[article_id] -= np.outer(mz, z.dot(m))/(1.0 + z.dot(mz))
        b_x[article_id] += y*z


def recommend(time, user_features, article_ids):
    global last_z
    global last_id
    ucb_max = -1
    x_max = np.zeros(N)
    id_max = np.random.choice(article_ids)
    z = np.array(user_features)

    for article_id in article_ids:
        if not M_x.has_key(article_id):
            M_x[article_id] = np.eye(N)
            b_x[article_id] = np.zeros(N)
        m = M_x[article_id]
        b = b_x[article_id]
        #m_inv = np.linalg.inv(m)
        # w = np.linalg.solve(m,b)
        w = m.dot(b)
        # ucb = w.dot(z) + ALPHA*np.sqrt(z.dot(np.linalg.solve(m, z)))
        ucb = w.dot(z) + ALPHA*np.sqrt(z.dot(m.dot(z)))
        if ucb_max < ucb:
            ucb_max = ucb
            id_max = article_id
    last_z = z
    last_id = id_max
    return id_max


# this is the version that will be called. Leave it as is
def reccomend(time, user_features, articles):
    return recommend(time, user_features, articles)

