(ns ^{:doc "implementation of the Convex Optimizer interface
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/ConvexOptimizer.html"}
    dl4clj.optimize.api.convex-optimizer
  (:import [org.deeplearning4j.optimize.api ConvexOptimizer]))

(defn get-batch-size
  "returns the batch size for the optimizer"
  [optim]
  (.batchSize optim))

(defn check-terminal-conditions?
  "Check termination conditions, sets up a search state

  :optim (optimizer), the optimizer

  :gradient (INDArray), layer gradients

  :old-score (double), the old score for the optimizer

  :score (double), the desired new score for the optimizer

  :iteration (int), what iteration the optimizer is on

  returns a boolean"
  [& {:keys [optim gradient old-score score iteration]}]
  (.checkTerminalConditions optim gradient old-score score iteration))

(defn get-conf
  "get the nn-conf associated with this optimizer"
  [optim]
  (.getConf optim))

(defn get-updater
  "returns the updater associated with this optimizer"
  [optim]
  (.getUpdater optim))

(defn get-gradient-and-score
  "the gradient and score for this optimizer"
  [optim]
  (.gradientAndScore optim))

(defn optimize
  "calls optimize! returns a boolean"
  [optim]
  (.optimize optim))

(defn post-step!
  "After the step has been made, do an action

  :line (INDArray) ... not sure what this is used for

  returns the optimizer"
  [& {:keys [optim line]}]
  (doto optim (.postStep line)))

(defn pre-process-line!
  "Pre preProcess a line before an iteration

  returns the optimizer"
  [optim]
  (doto optim (.preProcessLine)))

(defn get-score
  "The score for the optimizer so far"
  [optim]
  (.score optim))

(defn set-batch-size!
  "set the batch size for the optimizer

  :batch-size (int), the batch size

  returns the optimizer"
  [& {:keys [optim batch-size]}]
  (doto optim (.setBatchSize batch-size)))

(defn set-listeners!
  "sets the listeners for the supplied optimizer

  :listeners (collection), a collection of listeners
   - clojure data structures can be used

  returns the optimizer"
  [& {:keys [optim listeners]}]
  (doto optim (.setListeners listeners)))

(defn set-updater!
  "sets the updater for the optimizer

  :updater (updater), an updater to add to the optimizer

  returns the optimizer"
  [& {:keys [optim updater]}]
  (doto optim (.setUpdater updater)))

(defn set-up-search-state!
  "Based on the gradient and score, set up a search state

  :gradient (gradient), the gradient used to set up search state
   - see: dl4clj.nn.gradient.default-gradient

  :score (double), the score used to set up search state

  returns the optimizer

  THIS FN IS NOT DONE, NEED TO IMPLEMENT berkeley.pairs"
  [& {:keys [optim gradient score]}]
  (doto optim (.setupSearchState {gradient score})))

(defn update-gradient-according-to-params!
  "Update the gradient according to the configuration suc as adagrad, momentum and sparsity

  :gradient (gradient), see: dl4clj.nn.gradient.default-gradient

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  :batch-size (int), the batch size

  returns the optimizer"
  [& {:keys [optim gradient model batch-size]}]
  (doto optim (.updateGradientAccordingToParams gradient model batch-size)))
