(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier]))


(defn f1-score
  "With two arguments: Sets the input and labels and returns a score for the prediction wrt true labels.
  With three arguments: Returns the f1 score for the given examples."
  ([^Classifier this data]
   (.f1Score this data))
  ([^Classifier this examples labels]
   (.f1Score this examples labels)))

(defn fit
  "Fit the model"
  ([^Classifier this d]
   (.fit this d))
  ([^Classifier this examples labels]
   (.fit this examples labels)))

(defn label-probabilities
  "Returns the probabilities for each label for each example row wise"
  [^Classifier this examples]
  (.labelProbabilities this examples))

(defn num-labels
  "Returns the number of possible labels"
  [^Classifier this]
  (.numLabels this))

(defn predict
  "Takes in a list of examples For each row, returns a label"
  [^Classifier this examples]
  (.predict this examples))
