"""Size-to-weight regressors (fixed model set used for the augmentation comparison and the yield pipeline)."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class PowerLaw(BaseEstimator, RegressorMixin):
    """Allometric model m = a * x^b * y^c fitted by OLS in log space (with smearing correction)."""
    def fit(self, X, y):
        Z = np.log(np.clip(X, 1e-3, None)); self.lr_ = LinearRegression().fit(Z, np.log(np.clip(y, 1e-3, None)))
        r = np.log(np.clip(y, 1e-3, None)) - self.lr_.predict(Z); self.smear_ = float(np.mean(np.exp(r)))
        return self
    def predict(self, X):
        return np.exp(self.lr_.predict(np.log(np.clip(X, 1e-3, None)))) * self.smear_


def models():
    return {
        "Power law (allometric)": PowerLaw(),
        "Linear regression": LinearRegression(),
        "Poisson GLM": make_pipeline(StandardScaler(), PoissonRegressor(max_iter=5000)),
        "Gamma GLM": make_pipeline(StandardScaler(), GammaRegressor(max_iter=5000)),
        "K-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "Support vector regression": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.5)),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=4),
        "Extra-Trees": ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=4),
        "Gradient boosting": GradientBoostingRegressor(random_state=0),
    }



def swap_aug(X, y):
    return np.vstack([X, X[:, ::-1]]), np.concatenate([y, y])
