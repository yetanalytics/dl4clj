(ns dl4clj.optimize.api.optimizers
  (:import [org.deeplearning4j.optimize Solver]
           [org.deeplearning4j.optimize.solvers
            StochasticGradientDescent
            LineGradientDescent
            ConjugateGradient
            BaseOptimizer
            LBFGS
            BackTrackLineSearch])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Base optimizer methods not inherited from ConvexOptimizer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-default-step-fn-for-optimizer
  "returns the default step fn for a type of optimizer

  expects the class of the convex optimizer"
  [& {:keys [convex-optimizer-class as-code?]
      :or {as-code? true}}]
  (match [convex-optimizer-class]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(BaseOptimizer/getDefaultStepFunctionForOptimizer ~convex-optimizer-class))
         :else
         (BaseOptimizer/getDefaultStepFunctionForOptimizer convex-optimizer-class)))

(defn get-iteration-count
  "get the number of iterations the model has been through"
  [& {:keys [model as-code?]
      :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(BaseOptimizer/getIterationCount ~model))
         :else
         (BaseOptimizer/getIterationCount model)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Back track line search methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-max-iterations
  "returns the max number of iterations for the optimizer"
  [& {:keys [back-track as-code?]
      :or {as-code? true}}]
  (match [back-track]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMaxIterations ~back-track))
         :else
         (.getMaxIterations back-track)))

(defn get-step-max
  "returns the max value of the step fn"
  [& {:keys [back-track as-code?]
      :or {as-code? true}}]
  (match [back-track]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getStepMax ~back-track))
         :else
         (.getStepMax back-track)))

(defn set-abs-tolerance!
  "Sets the tolerance of absolute difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:back-track (_ :guard seq?)
           :tolerance (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~back-track (.setAbsTolx (double ~tolerance))))
         :else
         (doto back-track (.setAbsTolx tolerance))))

(defn set-max-iterations!
  "sets the max number of iterations for the optimizer

  :max-iterations (int), the value to set for the max iteration

  returns the back-track optimizer"
  [& {:keys [back-track max-iterations as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:back-track (_ :guard seq?)
           :max-iterations (:or (_ :guard number?)
                                (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~back-track (.setMaxIterations (int ~max-iterations))))
         :else
         (doto back-track (.setMaxIterations max-iterations))))

(defn set-relative-tolerance!
  "Sets the tolerance of relative difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:back-track (_ :guard seq?)
           :tolerance (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~back-track (.setRelTolx (double ~tolerance))))
         :else
         (doto back-track (.setRelTolx tolerance))))

(defn set-step-max!
  "sets the max step size for the back-track optimizer

  :step-max (double), the max value for the step size

  returns the back-track optimizer"
  [& {:keys [back-track step-max as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:back-track (_ :guard seq?)
           :step-max (:or (_ :guard number?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~back-track (.setStepMax (double ~step-max))))
         :else
         (doto back-track (.setStepMax step-max))))
