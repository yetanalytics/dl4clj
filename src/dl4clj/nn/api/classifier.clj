(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]
            [dl4clj.nn.api.model :refer :all]))


(defn f1-score
  "With two arguments (this and dataset): Sets the input and labels and returns a score for the prediction wrt true labels.
  With three arguments (this examples and labels): Returns the f1 score for the given examples."
  [& {:keys [this dataset examples labels]
      :as opts}]
  (cond-> this
    (contains-many? opts :this :dataset) (.f1Score dataset)
    (contains-many? opts :this :examples :labels) (.f1Score examples labels)))

(defn fit-classifier
  "Fit the model, supply either this and a data-set, this and a dataset-iterator
   or this, examples and labels"
  [& {:keys [this data-set dataset-iterator examples labels]
      :as opts}]
  (cond-> this
    (contains-many? opts :this :data-set) (.fit data-set)
    (contains-many? opts :this :dataset-iterator) (.fit dataset-iterator)
    (contains-many? opts :this :examples :labels) (.fit examples labels)))


(defn label-probabilities
  "Returns the probabilities for each label for each example row wise"
  [& {:keys [this examples]}]
  (.labelProbabilities this examples))

(defn num-labels
  "Returns the number of possible labels"
  [^Classifier this]
  (.numLabels this))

(defn predict
  "Takes in a list of examples For each row, returns a label
   or a datset of examples for each row, returns a label"
  [& {:keys [this examples dataset]
      :as opts}]
  (cond-> this
    (contains-many? opts :this :examples) (.predict examples)
    (contains-many? opts :this :dataset) (.predict dataset)))
