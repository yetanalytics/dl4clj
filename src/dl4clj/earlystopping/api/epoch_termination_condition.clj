(ns ^{:doc "Interface for termination conditions to be evaluated once per epoch (i.e., once per pass of the full data set), based on a score and epoch number

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/EpochTerminationCondition.html"}
    dl4clj.earlystopping.api.epoch-termination-condition
  (:import [org.deeplearning4j.earlystopping.termination EpochTerminationCondition]))

(defn initialize-epoch!
  "Initialize the epoch termination condition (often a no-op)

  returns the epoch termination condition"
  [epoch-term-cond]
  (doto epoch-term-cond (.initialize)))

(defn terminate-now-epoch?
  "Should the early stopping training terminate at this epoch,
  based on the calculated score and the epoch number?

  Returns true if training should terminated, or false otherwise

  :epoch-term-cond (epoch termination condition)
   - see dl4clj.earlystopping.termination-conditions
   - one of best-score-epoch-term-cond
            max-epochs-termination-condition
            score-improvement-epoch-termination-condition

  :epoch-n (int), Number of the last completed epoch (starting at 0)

  :score (double), Score calculated for this epoch"
  [& {:keys [epoch-term-cond epoch-n score]}]
  (.terminate epoch-term-cond epoch-n score))
