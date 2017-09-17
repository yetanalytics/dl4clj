(ns ^{:doc "Implementation of the methods found in the Classifier Interface.
fns are for classification models (this is for supervised learning)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier])
  (:require [dl4clj.utils :refer [contains-many? obj-or-code?]]
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
  [& {:keys [classifier dataset examples labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:classifier (_ :guard seq?)
           :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(.f1Score ~classifier ~dataset))
         [{:classifier _ :dataset _}]
         (.f1Score classifier dataset)
         [{:classifier (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.f1Score ~classifier
                    (vec-or-matrix->indarray ~examples)
                    (vec-or-matrix->indarray ~labels)))
         [{:classifier _ :examples _ :labels _}]
         (.f1Score classifier
                   (vec-or-matrix->indarray examples)
                   (vec-or-matrix->indarray labels))
         :else
         (assert false "you must supply a classifier and either a dataset or
examples and their labels")))

;; redundant, covered by fit! in model
#_(defn fit-classifier!
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
         [{:classifier (_ :guard seq?) :dataset (_ :guard seq?)}]
         `(doto ~classifier (.fit ~dataset))
         [{:classifier _ :dataset _}]
         (doto classifier (.fit dataset))
         [{:classifier (_ :guard seq?) :iter (_ :guard seq?)}]
         `(doto ~classifier (.fit ~iter))
         [{:classifier _ :iter _}]
         (doto classifier (.fit (reset-iterator! iter)))
         [{:classifier (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         `(doto ~classifier (.fit (vec-or-matrix->indarray ~examples)
                                  (vec-or-matrix->indarray ~labels)))
         [{:classifier _ :examples _ :labels _}]
         (doto classifier (.fit (vec-or-matrix->indarray examples)
                                (vec-or-matrix->indarray labels)))
         :else
         (assert false "you must supply a classifier and either a dataset,
 iterator obj, or examples and their labels")))

(defn label-probabilities
  "Returns the probabilities for each label for each example row wise

  :examples (INDArray or vec), the examples to classify (one example in each row)"
  [& {:keys [classifier examples as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:classifier (_ :guard seq?) :examples (:or (_ :guard vector?)
                                                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.labelProbabilities ~classifier (vec-or-matrix->indarray ~examples)))
         [{:classifier _ :examples _}]
         (.labelProbabilities classifier (vec-or-matrix->indarray examples))))

(defn num-labels
  "Returns the number of possible labels"
  [& {:keys [classifier as-code?]
      :or {as-code? true}}]
  (match [classifier]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numLabels ~classifier))
         :else
         (.numLabels classifier)))

(defn predict
  "Takes in a list of examples for each row (INDArray or vec), returns a label

   or

  takes a datset of examples for each row, returns a label"
  [& {:keys [classifier examples dataset as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:classifier (_ :guard seq?) :examples (:or (_ :guard vector?)
                                                      (_ :guard seq?))}]
         (obj-or-code? as-code? `(.predict ~classifier (vec-or-matrix->indarray ~examples)))
         [{:classifier _ :examples _}]
         (.predict classifier (vec-or-matrix->indarray examples))
         [{:classifier (_ :guard seq?) :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(.predict ~classifier ~dataset))
         [{:classifier _ :dataset _}]
         (.predict classifier dataset)
         :else
         (assert false "you must supply a classifier and either an INDArray of examples or a dataset")))
