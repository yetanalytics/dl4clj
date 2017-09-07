(ns ^{:doc "ScoreCalculator interface is used to calculate a score for a neural network. For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/ScoreCalculator.html"}
    dl4clj.earlystopping.api.score-calc
  (:import [org.deeplearning4j.earlystopping.scorecalc ScoreCalculator])
  (:require [clojure.core.match :refer [match]]))

(defn calculate-score
  "Calculate the score for the given MultiLayerNetwork

  :score-calculator (obj), the object that does the calculations
   - see: dl4clj.earlystopping.score-calc

  :mln (model), A Model is meant for predicting something from data.
   - a multi-layer-network
   - see: dl4clj.nn.multilayer.multi-layer-network and dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [score-calculator mln]
      :as opts}]
  ;; this also works for computation graphs but they have not been implemented yet
  (match [opts]
         [{:score-calculator (_ :guard seq?)
           :mln (_ :guard seq?)}]
         `(.calculateScore ~score-calculator ~mln)
         :else
         (.calculateScore score-calculator mln)))
