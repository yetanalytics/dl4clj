(ns dl4clj.earlystopping.early-stopping-config
  (:import [org.deeplearning4j.earlystopping EarlyStoppingConfiguration$Builder]
           [org.deeplearning4j.earlystopping.termination
            InvalidScoreIterationTerminationCondition
            MaxScoreIterationTerminationCondition
            MaxTimeIterationTerminationCondition
            ScoreImprovementEpochTerminationCondition
            BestScoreEpochTerminationCondition
            MaxEpochsTerminationCondition
            IterationTerminationCondition
            EpochTerminationCondition]
           [org.deeplearning4j.earlystopping.saver
            InMemoryModelSaver
            LocalFileGraphSaver
            LocalFileModelSaver]
           [org.deeplearning4j.earlystopping EarlyStoppingModelSaver]
           [org.deeplearning4j.earlystopping.scorecalc ScoreCalculator
            DataSetLossCalculator DataSetLossCalculatorCG]
           [java.nio.charset Charset]
           )
  (:require [dl4clj.constants :as enum]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethods for setting up an early stopping configuration
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti termination-condition generic-dispatching-fn)

(defmethod termination-condition :invalid-score-iteration [opts]
  (InvalidScoreIterationTerminationCondition.))

(defmethod termination-condition :max-score-iteration [opts]
  (let [conf (:max-score-iteration opts)
        max-score (:max-score conf)]
    (MaxScoreIterationTerminationCondition. max-score)))

(defmethod termination-condition :max-time-iteration [opts]
  (let [conf (:max-time-iteration opts)
        {max-time :max-time
         time-unit :time-unit} conf]
    (MaxTimeIterationTerminationCondition. max-time (enum/value-of
                                                     {:time-unit
                                                      time-unit}))))

(defmethod termination-condition :score-improvement-epoch [opts]
  (let [conf (:score-improvement-epoch opts)
        {max-no-improve :max-epochs-with-no-improvement
         min-improve :min-improvement} conf]
    (if (contains? conf :min-improvement)
      (ScoreImprovementEpochTerminationCondition. max-no-improve min-improve)
      (ScoreImprovementEpochTerminationCondition. max-no-improve))))

(defmethod termination-condition :best-score-epoch [opts]
  (let [conf (:best-score-epoch opts)
        {best-expected :best-expected-score
         lesser-better :is-less-better?} conf]
    (if (contains? conf :is-less-better?)
      (BestScoreEpochTerminationCondition. best-expected lesser-better)
      (BestScoreEpochTerminationCondition. best-expected))))

(defmethod termination-condition :max-epochs [opts]
  (let [conf (:max-epochs opts)
        max-es (:max conf)]
    (MaxEpochsTerminationCondition. max-es)))

(defmulti model-saver-type generic-dispatching-fn)

(defmethod model-saver-type :default [opts]
  (InMemoryModelSaver.))

(defmethod model-saver-type :in-memory [opts]
  (InMemoryModelSaver.))

(defmethod model-saver-type :local-file-graph [opts]
  (let [config (:local-file-graph opts)
        {dir :directory
         char-set :charset} config]
    (if (contains? config :charset)
      (LocalFileGraphSaver. dir char-set)
      (LocalFileGraphSaver. dir))))

(defmethod model-saver-type :local-file-model [opts]
  (let [config (:local-file-model opts)
        {dir :directory
         char-set :charset} config]
    (if (contains? config :charset)
      (LocalFileModelSaver. dir char-set)
      (LocalFileModelSaver. dir))))

(defmulti score-calc generic-dispatching-fn)
;; spark versions will be implemented later
;;https://deeplearning4j.org/doc/org/deeplearning4j/spark/earlystopping/SparkDataSetLossCalculator.html
;;https://deeplearning4j.org/doc/org/deeplearning4j/spark/earlystopping/SparkLossCalculatorComputationGraph.html

(defmethod score-calc :dataset-loss [opts]
  (let [conf (:dataset-loss opts)
        {iter :dataset-iterator
         avg? :average?} conf]
    (DataSetLossCalculator. iter avg?)))

(defmethod score-calc :dataset-loss-gc [opts]
  ;; this is for the computational graph type nn's
  ;; comp graph not implemented so this method should not be called until it is
  (let [conf (:dataset-loss opts)
        {iter :dataset-iterator
         avg? :average?} conf]
    (DataSetLossCalculatorCG. iter avg?)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fns for the main builder fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn epoch-term-cond
  [conds]
  (let [{score-improvement :score-improvement-epoch
         best-score :best-score-epoch
         max-n :max-epochs} conds
        score-term-cond (if (false? (nil? score-improvement))
                          (termination-condition {:score-improvement-epoch score-improvement}))
        best-score-cond (if (false? (nil? best-score))
                          (termination-condition {:best-score-epoch best-score}))
        max-n-cond (if (false? (nil? max-n))
                     (termination-condition {:max-epochs max-n}))
        conds (filterv #(not (nil? %)) (list score-term-cond best-score-cond max-n-cond))]
    (into-array EpochTerminationCondition conds)))

(defn iteration-term-cond
  [conds]
  (let [{invalid-score :invalid-score-iteration
         max-score :max-score-iteration
         max-time :max-time-iteration} conds
        invalid-score-cond (if (false? (nil? invalid-score))
                             (termination-condition {:invalid-score-iteration "zero args"}))
        max-score-cond (if (false? (nil? max-score))
                         (termination-condition {:max-score-iteration max-score}))
        max-time-cond (if (false? (nil? max-time))
                        (termination-condition {:max-time-iteration max-time}))
        conds (filterv #(not (nil? %)) (list invalid-score-cond max-score-cond max-time-cond))]
    (into-array IterationTerminationCondition conds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; the builder
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-early-stopping-config
  "Builder for setting up an early stopping configuration.  Mainly used during testing
  as a tool for neural network model building

  :epoch-termination-conditions (map), a map of desired epoch based termination conditions
   - possible conditions are: :score-improvement-epoch, :best-score-epoch and :max-epochs
    - configuration for the types of stopping conditions is detailed bellow
      - {:score-improvement-epoch {:max-epochs-with-no-improvement (int)
                                   :min-improvement (double)}}
      - {:best-score-epoch {:best-expected-score (double) :is-less-better? (boolean)}}
      - {:max-epochs {:max (int)}}
    - the :epoch-termination-conditions should contain atleast one of the maps above

  :iteration-termination-conditions (map), Termination conditions checked each iteration
   - configuration for the types of stopping conditions is detailed bellow
     - {:invalid-score-iteration {} (no opts necessary or used)}
     - {:max-score-iteration {:max-score (double)}}
     - {:max-time-iteration {:max-time (long) :time-unit (keyword)}}
       - :time-unit is one of :days, :hours, :minutes, :seconds, :microseconds,
         :milliseconds or :nanoseconds
   - the :iteration-termination-conditions should contain atleast one of the above maps

  :n-epochs (int), number of epochs to run before checking termination condition

  :model-saver (map), how the model is going to be saved (in memory vs to disk)
   - {:in-memory {} (no args)}
   - {:local-file-model {:directory (str) :charset (java character-set)}}
     - for avialable char sets, eval (Charset/availableCharsets)
     - char sets are optional

  :save-last-model? (boolean), wether or not to save the last model run

  :score-calculator (map), {:dataset-iterator (dataset-iterator) :average? (boolean)}
   - see dl4clj.datasets.datavec for how to create a dataset-iterator"
  [{:keys [epoch-termination-conditions n-epochs
           iteration-termination-conditions model-saver
           save-last-model? score-calculator]
      :as opts}]
  (.build
   (cond-> (EarlyStoppingConfiguration$Builder.)
     (contains? opts :epoch-termination-conditions)
     (.epochTerminationConditions (epoch-term-cond epoch-termination-conditions))
     (contains? opts :n-epochs)
     (.evaluateEveryNEpochs n-epochs)
     (contains? opts :iteration-termination-conditions)
     (.iterationTerminationConditions (iteration-term-cond iteration-termination-conditions))
     (contains? opts :n-epochs)
     (.evaluateEveryNEpochs n-epochs)
     (contains? opts :model-saver)
     (.modelSaver (first (into-array EarlyStoppingModelSaver [(model-saver-type model-saver)])))
     (contains? opts :save-last-model?)
     (.saveLastModel save-last-model?)
     (contains? opts :score-calculator)
     (.scoreCalculator (score-calc score-calculator)))))

(comment

  (new-early-stopping-config
   {:epoch-termination-conditions
    {:score-improvement-epoch {:max-epochs-with-no-improvement 2
                               :min-improvement 2.1}
     :best-score-epoch {:best-expected-score 20 :is-less-better? true}
     :max-epochs {:max 5}}
    :iteration-termination-conditions {:invalid-score-iteration {}
                                       :max-score-iteration {:max-score 10}
                                       :max-time-iteration {:max-time 20
                                                            :time-unit :minutes}}
    :model-saver {:local-file-model {:directory "temp/foo"}}
    :score-calculator {:dataset-loss
                       {:dataset-iterator
                        (dl4clj.datasets.datavec/iterator
                         {:rr-dataset-iter
                          {:record-reader
                           (datavec.api.records.readers/record-reader {:csv-rr {}})
                           :batch-size 5}})
                        :average? false}}})
  )
