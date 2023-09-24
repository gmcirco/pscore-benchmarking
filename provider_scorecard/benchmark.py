import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier


class Benchmark:
    """
    Compare providers on benchmarks using propensity scores

    Initalizes a propensity-score based benchmarking algorthim. Given some data, an 
    indicator for the focal group, and a set of predictor and evaluation features fits 
    a boosted regression model.

    Args:
        data: A pandas DataFrame containing the data.
        focal_indicator: A pandas Series indicating the focal group.
        predictor_features: A list of strings indicating the predictor features.
        evaluation_features: A list of strings indicating the evaluation features.

    Attributes:
        model: A fitted propensity score model.
        pscore: A pandas Series containing the propensity scores.
        wgt: A pandas Series containing the ATT weights.
        Xpred: A pandas DataFrame containing the predictor features.
        Xeval: A pandas DataFrame containing the evaluation features.
        outcomes: A list of estimated treatment effects for each outcome variable.
    """
    def __init__(self, data, focal_indicator, predictor_features, evaluation_features):
        self.data = data
        self.tr = focal_indicator
        self.pred_feat = predictor_features
        self.eval_feat = evaluation_features

        self.model = None
        self.pscore = None
        self.wgt = None

        # set up predictor and evaluation matrices
        self.Xpred = pd.get_dummies(self.data[self.pred_feat])
        self.Xeval = pd.get_dummies(self.data[self.eval_feat])

        # list to hold results
        self.outcomes = []

    def fit(self, lrate=0.1, nest=500):
        """
        Fit a propensity score model using a Gradient Boosting Classifier.

        Args:
            lrate: The learning rate of the model.
            nest: The number of estimators in the model.

        Returns:
            None.
        """

        # set model params
        self.lrate = lrate
        self.nest = nest

        self.model = GradientBoostingClassifier(
            learning_rate=self.lrate, n_estimators=self.nest
        )
        self.model.fit(self.Xpred, self.tr)

        # define propensity scores and att weights
        self.pscore = self.model.predict_proba(self.Xpred)[:, 1]
        self.wgt = self._wt_att(y=self.tr, score=self.pscore)

    def evaluate(self,digits=2):
        """
        Compute a difference-in-means estimate for the focal hospital.

        This function uses a linear regression model to estimate the treatment effect
        for the focal hospital. The model is fitted to the data from all hospitals,
        with the focal hospital being the treatment group and all other hospitals being
        the control group. The difference in the mean of the outcome variable between
        the treatment and control groups is the estimated treatment effect.

        Args:
            digits: The number of digits to round the results to.

        Returns:
            A list of estimated treatment effects for each outcome variable.
        """

        Xtemp = self.Xpred.copy()
        Xtemp['tr'] = self.tr

        for i in self.Xeval.columns:
            regr = LinearRegression()
            regr.fit(Xtemp, self.Xeval[i], sample_weight=self.wgt)

            # get coefficient for treatment
            res = round(regr.coef_[len(Xtemp.columns)-1],digits)
            self.outcomes.append(res)

        return self.outcomes

    def calc_balance(self):
        """
        Print balance statistics for all variables in the Xpred DataFrame.

        This function calls the `_balance()` function to calculate the weighted balance
        statistics for ATT weights. The results are printed to the console.

        Args:
            None.

        Returns:
            None.
        """

        # TODO: Store a pandas dataframe with balance statistics
        for v in self.Xpred.columns:
            bal = self._balance(var=v, X=self.Xpred, tr=self.tr, wgt=self.wgt)
            print(f"{v}:{bal}")

    def _wt_att(self, y, score):
        """
        Calculate Average Treatment Effect on the Treated (ATT) weights.

        Args:
            y: A Series indicating the treatment group.
            score: A Series of propensity scores.

        Returns:
            A Series of ATT weights.
        """
        weight = score / (1 - score)

        # define att weights
        wt = [1 if t == True else w for t, w in zip(y, weight)]

        return wt

    def _balance(self, var, X, tr, wgt, digits=2):
        """
        Compute weighted balance statistics for ATT weights

        Args:
            var: The name of the variable to be balanced.
            X: A DataFrame containing the data.
            tr: A boolean Series indicating the treatment group.
            wgt: A Series of weights.
            digits: The number of digits to round the results to.

        Returns:
            A tuple of two floats, representing the mean of the variable in the
            treatment group and the weighted mean of the variable in the control group.
        """

        tab = X[var]

        #
        tf, tc = tab[tr], tab[~tr]
        c_wgt = [v for v, c in zip(wgt, tr) if not c]

        # compute average
        mf = round(np.average(tf), digits)
        mc = round(np.average(tc, weights=c_wgt), digits)

        return (mf, mc)
