(ns dl4clj.optimize.api.optimizers
  (:import [org.deeplearning4j.optimize Solver]
           [org.deeplearning4j.optimize.solvers
            StochasticGradientDescent
            LineGradientDescent
            ConjugateGradient
            BaseOptimizer
            LBFGS
            BackTrackLineSearch]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Base optimizer methods not inherited from ConvexOptimizer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-default-step-fn-for-optimizer
  ;; seems to be the only useful fn in this ns
  ;; can be used in assisting model building
  "returns the default step fn for a type of optimizer"
  [convex-optimizer-class]
  (BaseOptimizer/getDefaultStepFunctionForOptimizer convex-optimizer-class))

(defn get-iteration-count
  "get the number of iterations the model has been through"
  [model]
  (BaseOptimizer/getIterationCount model))

(defn increment-iteration-count!
  ;; not a user facing fn
  ;; will be removed in core branch
  "increments the iteration count for a model by the specified amount

  :increment-by (int), the specified amount

  returns the mutated model"
  [& {:keys [model increment-by]}]
  (doto model (BaseOptimizer/incrementIterationCount increment-by)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Back track line search methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-max-iterations
  "returns the max number of iterations for the optimizer"
  [back-track]
  (.getMaxIterations back-track))

(defn get-step-max
  "returns the max value of the step fn"
  [back-track]
  (.getStepMax back-track))

(defn set-abs-tolerance!
  ;; not a user facing fn
  ;; will be removed in core branch
  "Sets the tolerance of absolute difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]}]
  (doto back-track (.setAbsTolx tolerance)))

(defn set-max-iterations!
  ;; not a user facing fn
  ;; will be removed in core branch
  "sets the max number of iterations for the optimizer

  :max-iterations (int), the value to set for the max iteration

  returns the back-track optimizer"
  [& {:keys [back-track max-iterations]}]
  (doto back-track (.setMaxIterations max-iterations)))

(defn set-relative-tolerance!
  ;; not a user facing fn
  ;; will be removed in core branch
  "Sets the tolerance of relative difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]}]
  (doto back-track (.setRelTolx tolerance)))

(defn set-score-for!
  ;; not a user facing fn
  ;; will be removed in the core branch
  "sets the score for the passed in params

  :params (INDArray), will need to test to write a good desc

  returns the back-track optimizer"
  [& {:keys [back-track params]}]
  (doto back-track (.setScoreFor params)))

(defn set-step-max!
  ;; not a user facing fn
  ;; will be removed in the core branch
  "sets the max step size for the back-track optimizer

  :step-max (double), the max value for the step size

  returns the back-track optimizer"
  [& {:keys [back-track step-max]}]
  (doto back-track (.setStepMax step-max)))
