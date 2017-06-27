(ns ^{:doc "Each iteration the listener is called, mainly used for debugging or visualizations
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/IterationListener.html"}
    dl4clj.optimize.api.iteration-listener
  (:import [org.deeplearning4j.optimize.api IterationListener]))

(defn invoke!
  ;; dont think this is a user facing fn
  ;; will be removed in core branch
  "changes the invoke field to true

  returns the listener"
  [listener]
  (doto listener (.invoke)))

(defn iteration-done!
  ;; dont think this is a user facing fn
  ;; will be removed in core branch
  "Event listener for each iteration

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  :iteration (int), the iteration

  returns a map containing the listener and model"
  [& {:keys [listener model iteration]}]
  (doto listener (.iterationDone model iteration))
  {:listener listener :model model})
