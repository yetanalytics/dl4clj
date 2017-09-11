(ns dl4clj.optimize.api.optimizers
  (:import [org.deeplearning4j.optimize Solver]
           [org.deeplearning4j.optimize.solvers
            StochasticGradientDescent
            LineGradientDescent
            ConjugateGradient
            BaseOptimizer
            LBFGS
            BackTrackLineSearch])
  (:require [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Base optimizer methods not inherited from ConvexOptimizer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-default-step-fn-for-optimizer
  "returns the default step fn for a type of optimizer

  expects the class of the convex optimizer"
  [convex-optimizer-class]
  (match [convex-optimizer-class]
         [(_ :guard seq?)]
         `(BaseOptimizer/getDefaultStepFunctionForOptimizer ~convex-optimizer-class)
         :else
         (BaseOptimizer/getDefaultStepFunctionForOptimizer convex-optimizer-class)))

(defn get-iteration-count
  "get the number of iterations the model has been through"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(BaseOptimizer/getIterationCount ~model)
         :else
         (BaseOptimizer/getIterationCount model)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Back track line search methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-max-iterations
  "returns the max number of iterations for the optimizer"
  [back-track]
  (match [back-track]
         [(_ :guard seq?)]
         `(.getMaxIterations ~back-track)
         :else
         (.getMaxIterations back-track)))

(defn get-step-max
  "returns the max value of the step fn"
  [back-track]
  (match [back-track]
         [(_ :guard seq?)]
         `(.getStepMax ~back-track)
         :else
         (.getStepMax back-track)))

(defn set-abs-tolerance!
  "Sets the tolerance of absolute difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]
      :as opts}]
  (match [opts]
         [{:back-track (_ :guard seq?)
           :tolerance (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(doto ~back-track (.setAbsTolx (double ~tolerance)))
         :else
         (doto back-track (.setAbsTolx tolerance))))

(defn set-max-iterations!
  "sets the max number of iterations for the optimizer

  :max-iterations (int), the value to set for the max iteration

  returns the back-track optimizer"
  [& {:keys [back-track max-iterations]
      :as opts}]
  (match [opts]
         [{:back-track (_ :guard seq?)
           :max-iterations (:or (_ :guard number?)
                                (_ :guard seq?))}]
         `(doto ~back-track (.setMaxIterations (int ~max-iterations)))
         :else
         (doto back-track (.setMaxIterations max-iterations))))

(defn set-relative-tolerance!
  "Sets the tolerance of relative difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]
      :as opts}]
  (match [opts]
         [{:back-track (_ :guard seq?)
           :tolerance (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(doto ~back-track (.setRelTolx (double ~tolerance)))
         :else
         (doto back-track (.setRelTolx tolerance))))

(defn set-step-max!
  "sets the max step size for the back-track optimizer

  :step-max (double), the max value for the step size

  returns the back-track optimizer"
  [& {:keys [back-track step-max]
      :as opts}]
  (match [opts]
         [{:back-track (_ :guard seq?)
           :step-max (:or (_ :guard number?)
                          (_ :guard seq?))}]
         `(doto ~back-track (.setStepMax (double ~step-max)))
         :else
         (doto back-track (.setStepMax step-max))))
