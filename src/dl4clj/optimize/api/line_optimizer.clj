(ns ^{:doc "Line optimizer interface adapted from mallet.
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/LineOptimizer.html"}
    dl4clj.optimize.api.line-optimizer
  (:import [org.deeplearning4j.optimize.api LineOptimizer]))

(defn optimize
  "line optimizer

  :params (INDArray), the parameters to optimize

  :gradient (INDArray), the gradient

  :search-direction (INDArray), the point/direction to go include

  returns the last step size used"
  [& {:keys [optim params gradient search-direction]}]
  (.optimize optim params gradient search-direction))
