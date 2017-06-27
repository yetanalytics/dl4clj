(ns ^{:doc "Interface for termination conditions to be evaluated once per iteration (i.e., once per minibatch). Used for example to more quickly terminate training, instead of waiting for an epoch to complete before checking termination conditions.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/IterationTerminationCondition.html"}
    dl4clj.earlystopping.api.iteration-termination-condition
  (:import [org.deeplearning4j.earlystopping.termination IterationTerminationCondition]))

(defn initialize-iteration!
  "Initialize the iteration termination condition (often a no-op)

  returns the epoch termination condition"
  [iter-term-cond]
  (doto iter-term-cond (.initialize)))

(defn terminate-now-iteration?
  "Should early stopping training terminate at this iteration,
  based on the score for the last iteration?

  return true if training should be terminated immediately, or false otherwise

  :iter-term-cond (iteration termination condition)
   - see dl4clj.earlystopping.termination-conditions
   - one of invalid-score-iteration-termination-condition
            max-score-iteration-termination-condition
            max-time-iteration-termination-condition

  :last-mini-batch-score (double), Score of the last minibatch"
  [& {:keys [iter-term-cond last-mini-batch-score]}]
  (.terminate iter-term-cond last-mini-batch-score))
