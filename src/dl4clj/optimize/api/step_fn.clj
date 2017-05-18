(ns ^{:doc "Custom step function for line search
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/StepFunction.html"}
    dl4clj.optimize.api.step-fn
  (:import [org.deeplearning4j.optimize.api StepFunction])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn step!
  "makes a step with the given params, the step-fn is returned

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns

  :features (INDArray), the input data

  :lines (INDArray), the line.... need to figure out what this does

  :step (double), the size of the step to make"
  [& {:keys [step-fn features line step]
      :as opts}]
  (cond (contains-many? opts :features :line :step)
        (doto step-fn (.step features line step))
        (contains-many? opts :features :line)
        (doto step-fn (.step features line))
        (contains? opts :step-fn)
        (doto step-fn (.step))
        :else
        (assert false "you must supply atleast a step function")))
