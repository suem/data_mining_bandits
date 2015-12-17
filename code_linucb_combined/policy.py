
import numpy as np

D = 6
ALPHA = 0.3

A_inv_dict = {}
b_dict = {}
theta_dict = {}
x_dict = {}

a = None
xu = None


def set_articles(articles):
    global A_inv_dict, b_dict, theta_dict, x_dict
    x_dict.update(articles)
    for k in x_dict:
        x_dict[k] = x_dict[k][3:]
    k = x_dict.keys()
    A_inv_dict.update(dict.fromkeys(k, np.eye(D)))
    b_dict.update(dict.fromkeys(k, np.zeros(D)))
    theta_dict.update(dict.fromkeys(k, np.zeros(D)))


def update(r):
    global A_inv_dict, b_dict, theta_dict, xu
    if not r == -1:
        A_inv = A_inv_dict[a]
        b = b_dict[a]
        tmp = A_inv.dot(xu)
        A_inv -= np.outer(tmp, xu.dot(A_inv))/(1.0 + xu.dot(tmp))
        b += r*xu
        theta_dict[a] = A_inv.dot(b)
        A_inv_dict[a] = A_inv
        b_dict[a] = b


def recommend(time, user_features, articles):
    global a, xu
    p_max = np.float('-inf')
    a_max = None
    xu_max = None
    u = user_features[3:]
    for art in articles:
        A_inv = A_inv_dict[art]
        theta = theta_dict[art]
        x = x_dict[art]
        x.extend(u)
        xu = np.array(x)
        p = theta.dot(xu) + ALPHA*np.sqrt(xu.dot(A_inv).dot(xu))
        if p > p_max:
            p_max = p
            a_max = art
            xu_max = xu
    xu = xu_max
    return a_max


def reccomend(time, user_features, articles):
    return recommend(time, user_features, articles)
