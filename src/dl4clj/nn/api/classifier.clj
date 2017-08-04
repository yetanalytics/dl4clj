(ns ^{:doc "Implementation of the methods found in the Classifier Interface.
fns are for classification models (this is for supervised learning)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.helpers :refer [reset-iterator!]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn f1-score
  "With two arguments (classifier and dataset):
   - Sets the input and labels and returns a score for the prediction.

  With three arguments (classifier, examples and labels):
   - Returns the f1 score for the given examples.
   - examples and labels should both be INDArrays or vectors
    - examples = the data you want to classify
    - labels = the correct classifcation for a a set of examples"
  [& {:keys [classifier dataset examples labels]
      :as opts}]
  (match [opts]
         [{:classifier _ :dataset _}]
         (.f1Score classifier dataset)
         [{:classifier _ :examples _ :labels _}]
         (.f1Score classifier
                   (vec-or-matrix->indarray examples)
                   (vec-or-matrix->indarray labels))
         :else
         (assert false "you must supply a classifier and either a dataset or
examples and their labels")))

(defn fit-classifier!
  "If dataset-iterator is supplied, trains the classifier based on the dataset-iterator
  if dataset or examples and labels are supplied, fits the classifier.

  :dataset = a dataset

  :iter (iterator), an iterator for going through a collection of dataset objects

  :examples = INDArray or vector of input data to be classified

  :labels = INDArray or vector of labels for the examples

  Returns the classifier after it has been fit"
  [& {:keys [classifier dataset iter examples labels]
      :as opts}]
  (match [opts]
         [{:classifier _ :dataset _}]
         (doto classifier (.fit dataset))
         [{:classifier _ :iter _}]
         (doto classifier (.fit (reset-iterator! iter)))
         [{:classifier _ :examples _ :labels _}]
         (doto classifier (.fit (vec-or-matrix->indarray examples)
                                (vec-or-matrix->indarray labels)))
         :else
         (assert false "you must supply a classifier and either a dataset,
 iterator obj, or examples and their labels")))

(defn label-probabilities
  "Returns the probabilities for each label for each example row wise

  :examples (INDArray or vec), the examples to classify (one example in each row)"
  [& {:keys [classifier examples]}]
  (.labelProbabilities classifier (vec-or-matrix->indarray examples)))

(defn num-labels
  "Returns the number of possible labels"
  [classifier]
  (.numLabels classifier))

(defn predict
  "Takes in a list of examples for each row (INDArray or vec), returns a label

   or

  takes a datset of examples for each row, returns a label"
  [& {:keys [classifier examples dataset]
      :as opts}]
  (match [opts]
         [{:classifier _ :examples _}]
         (.predict classifier (vec-or-matrix->indarray examples))
         [{:classifier _ :dataset _}]
         (.predict classifier dataset)
         :else
         (assert false "you must supply a classifier and either an INDArray of examples or a dataset")))
