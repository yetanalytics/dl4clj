(ns ^{:doc "Each iteration the listener is called, mainly used for debugging or visualizations
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/IterationListener.html"}
    dl4clj.optimize.api.iteration-listener
  (:import [org.deeplearning4j.optimize.api IterationListener]))

(defn invoke!
  "changes the invoke field to true

  returns the listener"
  [listener]
  (doto listener (.invoke)))

(defn invoked?
  "Was the listener invoked?"
  [listener]
  (.invoked listener))

(defn iteration-done!
  ;; not sure how to properly use this method
  "Event listener for each iteration

  :model (model), a built neural network
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :iteration (int), the iteration

  returns the listener"
  [& {:keys [listener model iteration]}]
  (doto listener (.iterationDone model iteration)))
