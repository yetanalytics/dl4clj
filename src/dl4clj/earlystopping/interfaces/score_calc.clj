(ns ^{:doc "ScoreCalculator interface is used to calculate a score for a neural network. For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/ScoreCalculator.html"}
    dl4clj.earlystopping.interfaces.score-calc
  (:import [org.deeplearning4j.earlystopping.scorecalc ScoreCalculator])
  (:require [dl4clj.earlystopping.score-calc :refer [score-calc]]))

(defn calculate-score
  "Calculate the score for the given MultiLayerNetwork

  :score-calcer (map or obj), the object that does the calculations or a config map to create the obj
   - one of: :data-set-loss-calc, :spark-data-set-loss-calc (not implemented)

   - see: dl4clj.earlystopping.score-calc

  :mln (model), A Model is meant for predicting something from data.
   - a multi-layer-network

   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [score-calcer model]}]
  ;; this also works for computation graphs but they have not been implemented yet
  (let [sc (if (map? score-calcer)
             (score-calc score-calcer)
             score-calcer)]
    (.calculateScore sc model)))
