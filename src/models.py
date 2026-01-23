import cloudpickle
from lifelines import WeibullAFTFitter
import pandas as pd
#from src.evaluate import evaluate_churn_predictions

class Weibull:

    def __init__(self, model: WeibullAFTFitter | None = None):
        self.model = model

    @classmethod
    def from_lifelines(cls, lifelines_model: WeibullAFTFitter) -> "Weibull":
        """
        Wrap a pretrained lifelines WeibullAFTFitter
        """
        return cls(model=lifelines_model)

    def fit(self, X: pd.DataFrame):
        model_params = {
            "penalizer": 0.01,
            "l1_ratio": 1.0,
        }
        self.model = WeibullAFTFitter(**model_params)
        self.model.fit(
            X,
            duration_col="T",
            event_col="E"
        )
        return self

    def predict(self, X: pd.DataFrame, horizon_days=(30, 60, 90)) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        X = X.copy()

        for horizon_day in horizon_days:
            col = f"p_churn_{horizon_day}_days"
            X[col] = self._survival_to_churn_proba(
                X=X,
                horizon_day=horizon_day,
            )

        return X
    
    def _survival_to_churn_proba(
        self,
        X: pd.DataFrame,
        horizon_day: int,
    ) -> pd.Series:
        """
        Returns P(churn within horizon_day)
        """
        surv_fn = self.model.predict_survival_function(X)

        probs = []
        for i in range(surv_fn.shape[1]):
            s = surv_fn.iloc[:, i]
            s_h = (
                s.loc[s.index <= horizon_day].iloc[-1]
                if (s.index <= horizon_day).any()
                else s.iloc[0]
            )
            probs.append(1 - s_h)

        return pd.Series(probs, index=X.index)
    
    def save(self) -> bytes:
        return cloudpickle.dumps(self)

    @staticmethod
    def load(blob: bytes) -> "Weibull":
        return cloudpickle.loads(blob)