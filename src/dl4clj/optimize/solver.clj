(ns ^{:doc "Generic purpose solver
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/Solver.html"}
    dl4clj.optimize.solver
  (:import [org.deeplearning4j.optimize Solver Solver$Builder]))

(defn build-solver
  "creates a solver for a given model with a nn-configuration and listeners.

  :nn-conf (conf), a neural network configuration.
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :single-listener (array), a listener object in an array.
   - see: dl4clj.optimize.listeners

  :multiple-listeners (collection), collection of listener objects
   - see: dl4clj.optimize.listeners

   - clojure data structures can be used

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  :build? (boolean), wether the builder should be built.
   - defaults to true

  if you supply a model, you must also supply a nn-conf otherwise get an NPE"
  [& {:keys [nn-conf single-listener multiple-listeners model build?]
      :or {build? true}
      :as opts}]
  ;; refactor to use builder fn
  (let [b (Solver$Builder.)]
    (assert (contains? opts :nn-conf) "you must supply a neural network configuration")
    (cond-> b
      (contains? opts :nn-conf) (.configure nn-conf)
      (contains? opts :single-listener) (.listener single-listener)
      (contains? opts :multiple-listeners) (.listeners multiple-listeners)
      (contains? opts :model) (.model model)
      (true? build?) (.build))))

(defn get-optimizer
  "returns the optimizer associated with the solver"
  [solver]
  (.getOptimizer solver))
