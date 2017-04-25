(ns ^{:doc "Implementation of the methods found in the Classifier Interface.
fns are for classification models (this is for supervised learning)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]
            [dl4clj.datasets.datavec :as iter]))

(defn f1-score
  "With two arguments (classifier and dataset):
   - Sets the input and labels and returns a score for the prediction.
  With three arguments (classifier, examples and labels):
   - Returns the f1 score for the given examples.
   - examples and labels should both be INDArrays
    - examples = the data you want to classify
    - labels = the correct classifcation for a a set of examples"
  [& {:keys [classifier dataset examples labels]
      :as opts}]
  (assert (or (contains-many? opts :classifier :dataset )
              (contains-many? opts :classifier :examples :labels))
          "you must supply a classifier and a dataset or a classifier, examples and labels")
  (cond (contains? opts :dataset)
        (.f1Score classifier dataset)
        (contains-many? opts :examples :labels)
        (.f1Score classifier examples labels)
        :else
        (assert false "you must supply a classifier and either a dataset or
examples and their labels")))

(defn fit-classifier!
  "If dataset-iterator is supplied, trains the classifier based on the dataset-iterator
  if dataset or examples and labels are supplied, fits the classifier.

  :data-set = a dataset
  :dataset-iterator {:iterator-type opts} (see dl4clj.datasets.datavec for more details)
  :examples = INDArray of input data to be classified
  :labels = INDArray or integer-array of labels for the examples

  Returns the classifier after it has been fit"
  [& {:keys [classifier data-set dataset-iterator examples labels]
      :as opts}]
  (cond (contains? opts :data-set)
        (.fit classifier data-set)
        (contains? opts :dataset-iterator)
        (.fit classifier (iter/iterator dataset-iterator))
        (contains-many? opts :examples :labels)
        (.fit classifier examples labels)
        :else
        (assert false "you must supply a classifier and either a dataset,
 dataset-iterator config map, or examples and their labels")))


(defn label-probabilities
  "Returns the probabilities for each label for each example row wise"
  [& {:keys [classifier examples]}]
  (.labelProbabilities classifier examples))

(defn num-labels
  "Returns the number of possible labels"
  [classifier]
  (.numLabels classifier))

(defn predict
  "Takes in a list of examples for each row (INDArray), returns a label
   or a datset of examples for each row, returns a label"
  [& {:keys [classifier examples dataset]
      :as opts}]
  (cond (contains? opts :examples) (.predict classifier examples)
        (contains? opts :dataset) (.predict classifier dataset)
        :else
        (assert false "you must supply a classifier and either an INDArray of examples or a dataset")))
