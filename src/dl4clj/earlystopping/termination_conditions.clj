(ns ^{:doc "ns for the creation of termination conditions. Both iteration and epoch termination conditions are created here."}
    dl4clj.earlystopping.termination-conditions
  (:import [org.deeplearning4j.earlystopping.termination
            InvalidScoreIterationTerminationCondition
            MaxScoreIterationTerminationCondition
            MaxTimeIterationTerminationCondition
            ScoreImprovementEpochTerminationCondition
            BestScoreEpochTerminationCondition
            MaxEpochsTerminationCondition
            IterationTerminationCondition
            EpochTerminationCondition])
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code?]]
            [dl4clj.helpers :refer [value-of-helper]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethods for calling the dl4j constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti termination-condition generic-dispatching-fn)

(defmethod termination-condition :invalid-score-iteration [opts]
  `(InvalidScoreIterationTerminationCondition.))

(defmethod termination-condition :max-score-iteration [opts]
  (let [conf (:max-score-iteration opts)
        max-score (:max-score conf)]
    `(MaxScoreIterationTerminationCondition. ~max-score)))

(defmethod termination-condition :max-time-iteration [opts]
  (let [conf (:max-time-iteration opts)
        {max-time :max-time-val
         time-unit :max-time-unit} conf]
    `(MaxTimeIterationTerminationCondition.
      ~(long max-time) ~(value-of-helper :time-unit time-unit))))

(defmethod termination-condition :score-improvement-epoch [opts]
  (let [conf (:score-improvement-epoch opts)
        {max-no-improve :max-n-epoch-no-improve
         min-improve :min-improve} conf]
    (if (contains? conf :min-improve)
      `(ScoreImprovementEpochTerminationCondition. ~max-no-improve ~min-improve)
      `(ScoreImprovementEpochTerminationCondition. ~max-no-improve))))

(defmethod termination-condition :best-score-epoch [opts]
  (let [conf (:best-score-epoch opts)
        {best-expected :best-expected-score
         lesser-better :is-less-better?} conf]
    (if (contains? conf :is-less-better?)
      `(BestScoreEpochTerminationCondition. ~best-expected ~lesser-better)
      `(BestScoreEpochTerminationCondition. ~best-expected))))

(defmethod termination-condition :max-epochs [opts]
  (let [conf (:max-epochs opts)
        max-n (:max-n conf)]
    `(MaxEpochsTerminationCondition. ~max-n)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns for creating termination conditions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-invalid-score-iteration-termination-condition
  "Terminate training at this iteration if score is NaN or Infinite for the last minibatch

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/InvalidScoreIterationTerminationCondition.html"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (termination-condition {:invalid-score-iteration {}})]
    (obj-or-code? as-code? code)))

(defn new-max-score-iteration-termination-condition
  "Iteration termination condition for terminating training if the minibatch score exceeds a certain value.
  This can occur for example with a poorly tuned (too high) learning rate

  :max-score (double), the max score value

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/MaxScoreIterationTerminationCondition.html"
  [& {:keys [max-score as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (termination-condition {:max-score-iteration opts})]
    (obj-or-code? as-code? code)))

(defn new-max-time-iteration-termination-condition
  "Terminate training based on max time.

  :max-time-val (number), what is the max amount of time allowed?

  :max-time-unit (keyword), the unit of :max-time-val
   - one of: :days, :hours, :minutes, :seconds, :microseconds, :milliseconds
             :nanoseconds

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/MaxTimeIterationTerminationCondition.html"
  [& {:keys [max-time-val max-time-unit as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (termination-condition {:max-time-iteration opts})]
    (obj-or-code? as-code? code)))

(defn new-score-improvement-epoch-termination-condition
  "Terminate training if best model score does not improve for N epochs

  :max-n-epoch-no-improve (int), the max number of consecutive epochs with no improvement

  :min-improve (double), the min amount of improvement required to return

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/ScoreImprovementEpochTerminationCondition.html"
  [& {:keys [max-n-epoch-no-improve min-improve as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (termination-condition {:score-improvement-epoch opts})]
    (obj-or-code? as-code? code)))

(defn new-best-score-epoch-termination-condition
  "Stop the training once we achieved an expected score.

  Normally this will stop if the current score is lower than the initialized score.

  If you want to stop the training once the score increases the defined score set :is-less-better? to false

  :best-expected-score (double), the target score we want to model to reach

  :less-is-better? (boolean), determines if we terminate on any increase in score
   - you may have underestimated how successful the model is at fitting the data

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/BestScoreEpochTerminationCondition.html"
  [& {:keys [best-expected-score less-is-better? as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (termination-condition {:best-score-epoch opts})]
    (obj-or-code? as-code? code)))

(defn new-max-epochs-termination-condition
  "Terminate training if the number of epochs exceeds the maximum number of epochs

  :max-n (int), the max number of allowed epochs

  :as-code? (boolean), will this fn return a java object or the code to create on

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/MaxEpochsTerminationCondition.html"
  [& {:keys [max-n as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (termination-condition {:max-epochs opts})]
    (obj-or-code? as-code? code)))
