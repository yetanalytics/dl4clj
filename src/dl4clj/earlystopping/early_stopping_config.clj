(ns ^{:doc "Early stopping configuration: Specifies the various configuration options for running training with early stopping.
Users need to specify the following:
(a) EarlyStoppingModelSaver: How models will be saved (to disk, to memory, etc) (Default: in memory)
(b) Termination conditions: at least one termination condition must be specified
(i) Iteration termination conditions: calculated once for each minibatch. For example, maxTime or invalid (NaN/infinite) scores
(ii) Epoch termination conditions: calculated once per epoch. For example, maxEpochs or no improvement for N epochs
(c) Score calculator: what score should be calculated at every epoch? (For example: test set loss or test set accuracy)
(d) How frequently (ever N epochs) should scores be calculated? (Default: every epoch)"}
    dl4clj.earlystopping.early-stopping-config
  (:import [org.deeplearning4j.earlystopping EarlyStoppingConfiguration$Builder]
           [org.deeplearning4j.earlystopping.termination
            IterationTerminationCondition
            EpochTerminationCondition]
           [org.deeplearning4j.earlystopping EarlyStoppingModelSaver])
  (:require [dl4clj.utils :refer [array-of builder-fn]]))

(defn new-early-stopping-config
  "Builder for setting up an early stopping configuration.  Mainly used during testing
  as a tool for neural network model building

  :epoch-termination-conditions (coll), a collection of epoch-termination conditions
   - possible conditions are: :score-improvement-epoch, :best-score-epoch and :max-epochs
    - see: dl4clj.earlystopping.termination-conditions

  :iteration-termination-conditions (coll), a collection of iteration termination conditions
   - possible conditions are: :invalid-score-iteration, :max-score-iteration, :max-time-iteration
     - see: dl4clj.earlystopping.termination-conditions

  :n-epochs (int), number of epochs to run before checking termination condition

  :model-saver (coll), how the model is going to be saved (in memory vs to disk)
   - see: dl4clj.earlystopping.model-saver
   - needs to be passed within a clojure data structure

  :save-last-model? (boolean), wether or not to save the last model run

  :score-calculator (score-calc), the calc of the error typically on a testing set
   - see:  dl4clj.earlystopping.score-calc

  :build? (boolean), build the configuration?
   - defaults to true

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingConfiguration.Builder.html"
  [& {:keys [epoch-termination-conditions n-epochs
             iteration-termination-conditions model-saver
             save-last-model? score-calculator build?]
      :or {build? true}
      :as opts}]
  (let [method-map {:epoch-termination-conditions     '.epochTerminationConditions
                    :n-epochs                         '.evaluateEveryNEpochs
                    :iteration-termination-conditions '.iterationTerminationConditions
                    :model-saver                      '.modelSaver
                    :save-last-model?                 '.saveLastModel
                    :score-calculator                 '.scoreCalculator}
        b `(EarlyStoppingConfiguration$Builder.)]
    ;; termination conditions need to be code
    ;; model saver needs to be code
    ;; score-calc needs to be code
   (cond-> (EarlyStoppingConfiguration$Builder.)
    (contains? opts :epoch-termination-conditions)
    (.epochTerminationConditions (array-of :data epoch-termination-conditions
                                           :java-type EpochTerminationCondition))
    (contains? opts :n-epochs)
    (.evaluateEveryNEpochs n-epochs)
    (contains? opts :iteration-termination-conditions)
    (.iterationTerminationConditions (array-of :data iteration-termination-conditions
                                               :java-type IterationTerminationCondition))
    (contains? opts :model-saver)
    (.modelSaver (first
                  (array-of :java-type EarlyStoppingModelSaver
                            :data model-saver)))
    (contains? opts :save-last-model?)
    (.saveLastModel save-last-model?)
    (contains? opts :score-calculator)
    (.scoreCalculator score-calculator)
    (true? build?)
    (.build))))
