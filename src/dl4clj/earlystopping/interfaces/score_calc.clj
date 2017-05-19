(ns ^{:doc "ScoreCalculator interface is used to calculate a score for a neural network. For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/ScoreCalculator.html"}
    dl4clj.earlystopping.interfaces.score-calc
  (:import [org.deeplearning4j.earlystopping.scorecalc ScoreCalculator]))

(defn calculate-score
  "Calculate the score for the given MultiLayerNetwork

  :score-calc (calculator), the object that does the calculations
   - one of: :data-set-loss-calc, :spark-data-set-loss-calc (not implemented)

   - see: TBD

  :mln (model), A Model is meant for predicting something from data.
   - a multi-layer-network

   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [score-calc model]}]
  ;; this also works for computation graphs but they have not been implemented yet
  (.calculateScore score-calc model))
