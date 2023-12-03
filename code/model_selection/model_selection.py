from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def perform_grid_search(model, param_grid, score_function, kf, X, y, n_jobs=None):
    score = make_scorer(score_function, greater_is_better=False)
    grid = GridSearchCV(model, param_grid=param_grid, cv=kf, scoring=score, n_jobs=n_jobs)

    grid.fit(X, y)

    return grid
